import torch
import torch.nn as nn
from typing import Dict, Union, List
from compression.quantization.quantized_spectral_layer import QuantizedSpectralConv
from compression.quantization.quantized_spectral_conv2dv2 import QuantizedSpectralConv2dV2
'''
# Helper function to compute the total memory footprint (in bytes) of a model.
def _get_model_size_in_bytes(model: nn.Module) -> int:
    total_size = 0
    # Sum the sizes of all parameters
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    # Sum the sizes of all buffers (e.g., our quantized int8 buffers)
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    return total_size
'''
class QuantizedLinear(nn.Module):
    '''
    Purpose:
    This class wraps an existing nn.Linear layer to “simulate” quantization.
    Explanation:
    The constructor takes an nn.Linear layer.
    It stores the number of input features, output features, and whether the layer uses a bias.
    '''
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.has_bias = linear.bias is not None


        # Compute scale from weight
        '''
        Purpose:
        Compute a scaling factor used to normalize the weight values.
        Explanation:
        The scale is the maximum absolute value in the weight tensor.
        If all values are zero (scale equals 0), we set it to 1.0 to avoid division by zero.
        '''
        # Instead of scaling by max(abs(weight)), scale to int8 range
        #changes on 7 march start from here
        # Compute min and max
        w_min = linear.weight.data.min().item()
        w_max = linear.weight.data.max().item()
        '''
        int8_max = 127  # Max range for int8
        self.scale = linear.weight.data.abs().max().item() / int8_max
        '''
        # Compute scale using (max - min) / 256
        #self.scale = (w_max - w_min) / 256
        self.scale = (w_max - w_min) / 65535

        if self.scale == 0:
            self.scale = 1.0
        


        # Quantize weight to int8 and register as buffer
        '''
        Purpose:
        Create a quantized version of the weight.
        Explanation:
        The original weight is divided by the scale, then rounded to the nearest integer.
        The result is clamped to the valid int8 range (-128 to 127) and converted to int8.
        It is stored as a buffer (using register_buffer) so that it does not require gradients.
        '''
        '''
        q_weight = (linear.weight.data / self.scale).round().clamp(-128, 127).to(torch.int8)
        '''
        # Quantize weight
        #q_weight = torch.round((linear.weight.data - w_min) / self.scale - 127).clamp(-127, 127).to(torch.int8)
        q_weight = torch.round((linear.weight.data - w_min) / self.scale - 32768).clamp(-32768, 32767).to(torch.int16)
        self.register_buffer('q_weight', q_weight)

        # Store min for dequantization
        self.register_buffer('w_min', torch.tensor(w_min, dtype=torch.float32))

        '''
        Purpose:
        Quantize the bias (if present) in the same way as the weight.
        Explanation:
        Computes a separate scale for the bias.
        Quantizes and registers the bias as a buffer. If no bias exists, registers None.
        '''
        if self.has_bias:
            '''
            self.bias_scale = linear.bias.data.abs().max().item()
            '''

            b_min = linear.bias.data.min().item()
            b_max = linear.bias.data.max().item()
            #self.bias_scale = (b_max - b_min) / 256
            self.bias_scale = (b_max - b_min) / 65535

            if self.bias_scale == 0:
                self.bias_scale = 1.0
            '''
            q_bias = (linear.bias.data / self.bias_scale).round().clamp(-128, 127).to(torch.int8)
            '''
            #q_bias = torch.round((linear.bias.data - b_min) / self.bias_scale - 127).clamp(-127, 127).to(torch.int8)
            q_bias = torch.round((linear.bias.data - b_min) / self.bias_scale - 32768).clamp(-32768, 32767).to(torch.int16)
            self.register_buffer('q_bias', q_bias)
            self.register_buffer('b_min', torch.tensor(b_min, dtype=torch.float32))
        else:
            self.register_buffer('q_bias', None)



    '''
    Purpose:
    During the forward pass, convert the stored int8 quantized buffers back to float.
    Explanation:
    The quantized weight (and bias) are converted to float and multiplied by the 
    respective scale to recover an approximation of the original values.
    The standard linear function is then applied using these dequantized weights.
    '''
    def forward(self, x):
        # Dequantize weight and bias on-the-fly
        '''
        weight = self.q_weight.float() * self.scale
        bias = self.q_bias.float() * self.bias_scale if self.has_bias else None
        '''
        # Dequantize weight
        #weight = (self.q_weight.float() + 127) * self.scale + self.w_min
        weight = (self.q_weight.float() + 32768) * self.scale + self.w_min
        #bias = (self.q_bias.float() + 127) * self.bias_scale + self.b_min if self.has_bias else None
        bias = (self.q_bias.float() + 32768) * self.bias_scale + self.b_min if self.has_bias else None
        return nn.functional.linear(x, weight, bias)



class QuantizedConv1d(nn.Module):
    '''
    Purpose:
    A wrapper for a Conv1d layer that is designed for 1*1 convolutions.
    Explanation:
    Checks that the kernel size is exactly 1; otherwise, quantization isn't supported.

    '''
    def __init__(self, conv: nn.Conv1d):
        super().__init__()
        if conv.kernel_size != (1,):
            raise ValueError("Only Conv1d with kernel_size=1 is supported")


        '''
        Purpose:
        Copy over all relevant convolution parameters.
        Explanation:
        Stores attributes like in_channels, out_channels, stride, etc., from the original convolution.
        '''    
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.has_bias = conv.bias is not None

        '''
        Purpose:
        Quantize the convolution weight.
        Explanation:
        Similar to QuantizedLinear: computes a scale factor and quantizes the weight to int8, storing it as a buffer.
        '''
        '''
        int8_max = 127  # Max range for int8
        self.scale = conv.weight.data.abs().max().item() / int8_max
        '''
        w_min = conv.weight.data.min().item()
        w_max = conv.weight.data.max().item()

        # Ensure the scale is never zero (to avoid division errors)
        #self.scale = (w_max - w_min) / 256
        self.scale = (w_max - w_min) / 65535
        if self.scale == 0:
            self.scale = 1.0
        '''    
        q_weight = (conv.weight.data / self.scale).round().clamp(-128, 127).to(torch.int8)
        '''
        #q_weight = torch.round((conv.weight.data - w_min) / self.scale - 127).clamp(-127, 127).to(torch.int8)
        q_weight = torch.round((conv.weight.data - w_min) / self.scale - 32768).clamp(-32768, 32768).to(torch.int16)
        self.register_buffer('q_weight', q_weight)
        self.register_buffer('w_min', torch.tensor(w_min, dtype=torch.float32))

        '''
        Purpose:
        Quantize and store the bias (if available) in a similar manner.
        '''
        if self.has_bias:
            '''
            self.bias_scale = conv.bias.data.abs().max().item()
            '''
            b_min = conv.bias.data.min().item()
            b_max = conv.bias.data.max().item()
            #self.bias_scale = (b_max - b_min) / 256
            self.bias_scale = (b_max - b_min) / 65535

            if self.bias_scale == 0:
                self.bias_scale = 1.0
            '''
            q_bias = (conv.bias.data / self.bias_scale).round().clamp(-128, 127).to(torch.int8)
            '''
            #q_bias = torch.round((conv.bias.data - b_min) / self.bias_scale - 127).clamp(-127, 127).to(torch.int8)
            q_bias = torch.round((conv.bias.data - b_min) / self.bias_scale - 32768).clamp(-32768, 32767).to(torch.int16)
            self.register_buffer('q_bias', q_bias)
            self.register_buffer('b_min', torch.tensor(b_min, dtype=torch.float32))
        else:
            self.register_buffer('q_bias', None)
    


    '''
    Purpose:
    Dequantize the weight and bias during the forward pass.
    Explanation:
    Converts the stored int8 buffers back to float by multiplying with the scale.
    Performs the convolution using the dequantized values.
    '''
    def forward(self, x):
        '''
        weight = self.q_weight.float() * self.scale
        bias = self.q_bias.float() * self.bias_scale if self.has_bias else None
        '''
        # Dequantize weight
        # weight = (self.q_weight.float() + 127) * self.scale + self.w_min
        # bias = (self.q_bias.float() + 127) * self.bias_scale + self.b_min if self.has_bias else None
        weight = (self.q_weight.float() + 32768) * self.scale + self.w_min
        bias = (self.q_bias.float() + 32768) * self.bias_scale + self.b_min if self.has_bias else None

        return nn.functional.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)



'''
Purpose:
This class performs dynamic quantization on the model by replacing eligible layers with their quantized versions.
'''
class DynamicQuantization:
    """
    Custom dynamic quantization that replaces eligible layers (nn.Linear and Conv1d with kernel_size=1)
    with quantized wrappers. For SpectralConv layers, if the weights are complex we skip quantization.
    """
    def __init__(self, model: nn.Module):
        '''
        Explanation:
        Stores the model to be compressed.
        Initializes a dictionary to track which layers have been replaced with quantized wrappers.
        '''
        self.model = model
        self.compressed_layers = {}  # Maps layer name to quantized module


    '''
    Purpose:
    Replace an nn.Linear layer with a QuantizedLinear wrapper.
    Explanation:
    Creates the wrapper, stores it in compressed_layers, and returns it.
    '''
    def compress_FC(self, layer: nn.Linear, name: str):
        quantized_linear = QuantizedLinear(layer)
        self.compressed_layers[name] = quantized_linear
        return quantized_linear


    '''
    Purpose:
Replace a 1*1 nn.Conv1d layer with a QuantizedConv1d wrapper.
Explanation:
Similar to compress_FC but for Conv1d layers.

    '''
    def compress_conv1d(self, layer: nn.Conv1d, name: str):
        if layer.kernel_size != (1,):
            raise ValueError(f"Layer {name} is Conv1d but kernel_size != 1. Skipping quantization.")
        quantized_conv = QuantizedConv1d(layer)
        self.compressed_layers[name] = quantized_conv
        return quantized_conv
    

    '''
    Purpose:
Process a SpectralConv layer if possible.
Explanation:
First, check if the layer's weight can be converted to a dense tensor via to_tensor().
If the weight is complex, print a warning and skip quantization.
Otherwise, quantize the weight (using quantize_tensor), then immediately dequantize it (multiply back by the scale).
Replace the weight using from_tensor() if available and store an identifier in compressed_layers.
    '''

    
    def compress_spectral_conv(self, layer, name: str):
        """
        Replaces the given SpectralConv `layer` with a QuantizedSpectralConv.
        """
        # Potential checks:
        # skip if layer is complex_data= True? or not
        # skip if you want to handle factorization in a certain way?
        
        # Just create the wrapper:
        #print(f"[Debug] Inside compress_spectral_conv for {name}!")
        quantized_spectral = QuantizedSpectralConv(
            spectral_layer=layer,
            in_channels=layer.in_channels,
            out_channels=layer.out_channels,
            n_modes=layer.n_modes,
            complex_data=layer.complex_data if hasattr(layer, "complex_data") else False,
            max_n_modes=layer.max_n_modes if hasattr(layer, "max_n_modes") else None,
            bias=layer.bias is not None,
            separable=layer.separable,
            resolution_scaling_factor=layer.resolution_scaling_factor,
            fno_block_precision=layer.fno_block_precision if hasattr(layer, "fno_block_precision") else "full",
            rank=layer.rank if hasattr(layer, "rank") else 0.5,
            factorization=layer.factorization,
            implementation=layer.implementation,
            fixed_rank_modes=layer.fixed_rank_modes if hasattr(layer, "fixed_rank_modes") else False,
            decomposition_kwargs=layer.decomposition_kwargs if hasattr(layer, "decomposition_kwargs") else {},
            init_std=layer.init_std if hasattr(layer, "init_std") else "auto",
            fft_norm=layer.fft_norm,
            device=layer.weight.to_tensor().device if hasattr(layer.weight, "to_tensor") else None,
        )
        self.compressed_layers[name] = quantized_spectral
        return quantized_spectral

    def compress_spectral_conv2dV2(self, layer, name: str):
        """
        Replaces the given SpectralConv2dV2 layer with a QuantizedSpectralConv2dV2.
        """
        quantized_spectral_2dV2 = QuantizedSpectralConv2dV2(layer)
        self.compressed_layers[name] = quantized_spectral_2dV2
        return quantized_spectral_2dV2
    




    '''
    Purpose:
A helper function to quantize any given tensor.
Explanation:
Computes the maximum absolute value (scale).
Divides the tensor by the scale, rounds, clamps to the int8 range, and converts to int8.
Returns both the quantized tensor and the scale.
    '''
    def quantize_tensor(self, tensor: torch.Tensor) -> (torch.Tensor, float):
        scale = tensor.abs().max().item()
        if scale == 0:
            scale = 1.0
        q_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return q_tensor, scale
    


    '''
    Purpose:
The main method to apply dynamic quantization across the model.
Explanation:
Computes the total number of parameters (used later for stats).
Iterates over each module (using named_modules()).
Depending on the type of the module, calls the corresponding compression function.
Replaces the original module in the model with the quantized version using replace_module.
Returns the modified model.
    '''
    def compress(self) -> nn.Module:
        '''
        self.original_params = sum(p.numel() for p in self.model.parameters())
        '''
        # 1) Store float size (model before compression)
        self.size_before = self._get_model_size_in_bytes(self.model)

        # Iterate over model modules and replace eligible ones with quantized wrappers.
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                quantized_module = self.compress_FC(module, name)
                self.replace_module(name, quantized_module)
            elif isinstance(module, nn.Conv1d) and module.kernel_size == (1,):
                quantized_module = self.compress_conv1d(module, name)
                self.replace_module(name, quantized_module)   
            elif type(module).__name__ == "SpectralConv":
                new_module = self.compress_spectral_conv(module, name)
                self.replace_module(name, new_module)
            elif type(module).__name__ == "SpectralConv2dV2":
                new_module = self.compress_spectral_conv2dV2(module, name)
                self.replace_module(name, new_module)
            
        print("------------------------------[Dynamic Quantization] Compression applied successfully------------------------------")        
        return self.model
        
    

    '''
    Purpose:
Replace a module in the model given its name (e.g., "fno_blocks.convs.0").
Explanation:
Splits the module name by '.' to traverse the model's attribute tree.
Retrieves the parent module and sets the attribute corresponding to the final part to the new module.
    '''
    def replace_module(self, module_name: str, new_module: nn.Module):
        parts = module_name.split('.')
        parent = self.model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_module)
    

    '''
    Purpose:
Provide a summary of the compression results.
Explanation:
Computes the original model size in bytes (assuming each parameter is float32, i.e., 4 bytes).
Computes a hypothetical quantized size if parameters were stored as int8 (1 byte each).
Calculates a compression ratio and a simulated "sparsity" (1 - compression ratio).
Returns these values along with the names of the compressed layers.
    '''

    '''
    def get_compression_stats(self) -> Dict[str, Union[int, float, List[str]]]:
        # Assume float32: 4 bytes per parameter.
        original_size = self.original_params * 4
        # Hypothetical quantized size if stored as int8: 1 byte per parameter.
        quantized_size = self.original_params * 1
        compression_ratio = quantized_size / original_size
        sparsity = 1 - compression_ratio  # Note: this is only a simulated value.
        return {
            "original_size": original_size,
            "quantized_size": quantized_size,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "compressed_layers": list(self.compressed_layers.keys()),
        }
    '''

    def get_compression_stats(self):
        # 3) After compression, measure again
        size_after = self._get_model_size_in_bytes(self.model)
        ratio = size_after / self.size_before
        return {
            "original_size": self.size_before,
            "quantized_size": size_after,
            "compression_ratio": ratio,
            "sparsity": 1 - ratio,
            "dyquantized_layers": list(self.compressed_layers.keys()),
        }

    def _get_model_size_in_bytes(self, model):
        total_size = 0
        for p in model.parameters():
            total_size += p.nelement() * p.element_size()
        for b in model.buffers():
            total_size += b.nelement() * b.element_size()
        return total_size

