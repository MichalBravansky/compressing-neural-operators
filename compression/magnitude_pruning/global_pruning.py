from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from neuralop.models import FNO
from neuralop.losses import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
import numpy as np
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training.training_state import load_training_state
import torch
import torch.nn as nn
from typing import Dict
from compression.base import CompressionTechnique


class GlobalMagnitudePruning(CompressionTechnique):
    """Global magnitude-based pruning that operates on all submodules 
    which have a '.weight' attribute (including factorized or standard).
    
    Parameters
    ----------
    model : nn.Module
        The model to be pruned
    prune_ratio : float, optional
        Fraction of parameters to zero out, by default 0.2
    """
    def __init__(self, model: nn.Module, prune_ratio: float = 0.2):
        self.model = model
        self.prune_ratio = prune_ratio
        self.masks: Dict[str, torch.Tensor] = {}
        self.modules_to_prune = []

    def _collect_all_weights(self):
        """Traverse all submodules. If a submodule has a '.weight', we gather it 
        (reconstructing if factorized).
        
        Returns
        -------
        param_data : list
            List of flattened weight Tensors
        param_shapes : dict
            Dictionary mapping names to original shapes
        """
        param_data = []
        param_shapes = {}

        for full_name, module in self.model.named_modules():
            if not hasattr(module, "weight"):
                continue
            if module.weight is None:
                continue

            if hasattr(module.weight, "to_tensor"):
                w = module.weight.to_tensor()
            else:
                w = module.weight

            if not hasattr(w, "reshape"):
                continue

            param_data.append(w.reshape(-1))
            param_shapes[full_name] = w.shape

            self.modules_to_prune.append((full_name, module))

        return param_data, param_shapes

    def compute_masks(self) -> None:
        """Compute one global threshold over all layers' weights, 
        then build a binary mask for each layer.
        """
        param_data, param_shapes = self._collect_all_weights()

        all_weights = torch.cat(param_data)
        total_elems = all_weights.numel()

        cutoff_idx = int(self.prune_ratio * total_elems)
        if cutoff_idx == 0:
            global_threshold = float("-inf")
        else:
            sorted_weights = all_weights.abs().sort().values
            global_threshold = sorted_weights[cutoff_idx].item()

        idx_start = 0
        self.masks = {}

        for full_name, module in self.modules_to_prune:
            if hasattr(module.weight, "to_tensor"):
                w = module.weight.to_tensor()
            else:
                w = module.weight

            if not hasattr(w, "reshape"):
                continue

            mask = (w.abs() > global_threshold).float()
            self.masks[full_name] = mask

    def apply_masks(self) -> None:
        """Apply the computed masks to each submodule's weight."""
        for full_name, module in self.modules_to_prune:
            if hasattr(module.weight, "to_tensor"):
                w = module.weight.to_tensor()
            else:
                w = module.weight

            mask = self.masks[full_name]
            masked_weight = w * mask

            if hasattr(module.weight, "from_tensor"):
                module.weight.from_tensor(masked_weight)
            else:
                module.weight.data.copy_(masked_weight)

    def compress(self) -> None:
        """Execute the full pipeline: compute masks, then apply them."""
        self.compute_masks()
        self.apply_masks()

    def get_sparsity(self) -> float:
        """Returns fraction of zero entries among all pruned weights.
        
        Returns
        -------
        float
            Sparsity ratio between 0 and 1
        """
        total_params = 0
        zero_params = 0
        for mask in self.masks.values():
            total_params += mask.numel()
            zero_params += (mask == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0

    def get_compression_stats(self) -> Dict[str, float]:
        """Get statistics about the compression
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing compression metrics
        """
        return {"sparsity": self.get_sparsity()}