from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Dict
from copy import deepcopy

class CompressionTechnique(ABC):
    """Base class for implementing model compression techniques
    
    Parameters
    ----------
    model : nn.Module
        The model to be compressed
    """
    def __init__(self, model: nn.Module):
        self.model = model
    
    @abstractmethod
    def compress(self) -> None:
        """Apply the compression technique to the model"""
        pass
    
    @abstractmethod
    def get_compression_stats(self) -> Dict[str, float]:
        """Get statistics about the compression"""
        pass

class CompressedModel(nn.Module):
    """Wrapper class that adds compression capabilities to any model
    
    Parameters
    ----------
    model : nn.Module
        The model to be compressed
    compression_technique : CompressionTechnique
        The compression technique to apply
    create_replica : bool, default=False
        If True, creates a deep copy of the model before compression
    """
    def __init__(self, model: nn.Module, compression_technique, create_replica: bool = False):
        super().__init__()
        if create_replica:
            original_device = next(model.parameters()).device
            cpu_model = model.cpu()
            self.model = deepcopy(cpu_model).to(original_device)
            model.to(original_device)
        else:
            self.model = model
            
        self.compression = compression_technique(self.model)
        self.compression.compress()
    
    def get_compression_stats(self) -> Dict[str, float]:
        return self.compression.get_compression_stats()
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)