import torch
import torch.ao.quantization
import torch.nn as nn
from compression.base import CompressionTechnique
from typing import Dict
import math
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.coda_layer import CODALayer
from .observer_classes import *
from functools import partial
from .quantised_forwards import *

class UniformQuantisation(CompressionTechnique):
    def __init__(self, model: nn.Module, num_bits: int = 8, num_calibration_runs: int = 16):
        super().__init__(model)
        self.model = model
        self.num_bits = num_bits
        self.num_calibration_runs = num_calibration_runs
        if math.log2(num_bits) % 1 != 0:
            raise ValueError("Number of bits must be a power of 2")
        if num_bits == 8 or num_bits == 32:
            self.type = (lambda bits: getattr(torch, f"qint{bits}", None))(self.num_bits)
        else:
            self.type = (lambda bits: getattr(torch, f"int{bits}", None))(self.num_bits)
        self.init_size = self.get_size()

    def compress(self) -> None:
        self._quantise_model()
    
    def get_compression_stats(self) -> Dict[str, float]:
        return {"compression_ratio": self.get_size()/self.init_size,
                "bits": self.num_bits,
                "original_size": self.init_size,
                "compressed_size": self.get_size(),
                "sparsity": 1-(self.get_size()/self.init_size),
                }

    def _quantise_model(self) -> None:
        model_device = next(self.model.parameters()).device
        self.model.eval()
        self.model.cpu()
        model_name = self.model._get_name()
        quantise_methods = {
            "FNO": self._quantise_fno,
            "DeepONet": self._quantise_deeponet,
            "CODANO": self._quantise_codano
        }
        quantise_method = quantise_methods.get(model_name)
        if quantise_method:
            quantise_method()
        else:
            raise ValueError(f"Quantization method for model {model_name} is not defined")
        self.model.to(model_device)
        
    def _quantise_fno(self) -> None:
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        custom_class_observers = {"float_to_observed_custom_module_class": {SpectralConv: SpectralConvObserverCompatible}}
        torch.ao.quantization.prepare(self.model, inplace=True, prepare_custom_config_dict=custom_class_observers)

        for i in range(self.num_calibration_runs):
            input_tensor = torch.randn(1, 1, 16, 16)
            self.model(input_tensor)

        torch.ao.quantization.convert(self.model, inplace=True)

        self.model.forward = partial(quantised_fno_forward, self.model)
        self.model.fno_blocks.forward_with_postactivation = partial(quantised_forward_with_postactivation, self.model.fno_blocks)
        self.model.fno_blocks.forward_with_preactivation = partial(quantised_forward_with_preactivation, self.model.fno_blocks)

    def _quantise_deeponet(self) -> None:
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        torch.ao.quantization.prepare(self.model, inplace=True)

        for i in range(self.num_calibration_runs):
            input_tensor = torch.randn(1, 1, 128, 128)
            input_tensor2 = torch.randn(1, 1, 128, 128)
            self.model(input_tensor, input_tensor2)

        torch.ao.quantization.convert(self.model, inplace=True)

        self.model.forward = partial(quantised_deeponet_forward, self.model)

    def _quantise_codano(self):
        self.model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        custom_class_observers = {"float_to_observed_custom_module_class": {SpectralConv: SpectralConvObserverCompatible}}
        torch.ao.quantization.prepare(self.model, 
                                      inplace=True, 
                                      prepare_custom_config_dict=custom_class_observers
                                      )
        for i in range(self.num_calibration_runs):
            input_tensor = torch.randn(3, 1, 1024, 1024)
            if self.model.static_channel_dim > 0:
                static_channel = torch.randn(1, self.model.static_channel_dim, 32, 32)
                self.model(in_data=input_tensor, static_channel=static_channel, variable_ids=self.model.variable_ids)
                continue
            self.model(in_data=input_tensor, variable_ids=self.model.variable_ids)

        torch.ao.quantization.convert(self.model, inplace=True)

        for name, module in self.model.named_modules():
            if isinstance(module, CODALayer):
                module._forward_equivariant = partial(quantised_docano_coda_layer_forward_equivariant, module)
                module.compute_attention = partial(quantised_compute_attention, module)
            if isinstance(module, FNOBlocks):
                module.forward_with_postactivation = partial(quantised_forward_with_postactivation_docano, module)
                module.forward_with_preactivation = partial(quantised_forward_with_preactivation_docano, module)

        self.model.forward = partial(quantised_docano_forward, self.model)
        

    def _get_parent_module(self, module_name: str):
        """
        Helper function to get the parent module of a given module name.
        """
        module_names = module_name.split('.')
        parent_module = self.model
        for name in module_names[:-1]:
            parent_module = getattr(parent_module, name)
        return parent_module
    
    def get_size(self) -> float:
        total_size = 0
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        total_size = param_size + buffer_size
        return total_size