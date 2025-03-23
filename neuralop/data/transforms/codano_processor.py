import torch
from torch import nn

class CODANODataProcessor(nn.Module):
    def __init__(self, in_normalizer=None, out_normalizer=None, device='cuda'):
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = device
    
    def preprocess(self, batch):
        # Extract x and y from the batch
        x = batch.get('x')
        y = batch.get('y')
        
        # Normalize if normalizers exist
        if self.in_normalizer is not None:
            x = self.in_normalizer.encode(x)
        if self.out_normalizer is not None:
            y = self.out_normalizer.encode(y)
            
        # Return in format expected by CODANO
        return {
            'in_data': x.to(self.device),
            'static_channels': None,
            'variable_ids': ["a1"],
            'y': y.to(self.device)
        }
    
    def postprocess(self, out, sample):
        if self.out_normalizer is not None:
            out = self.out_normalizer.decode(out)
        return out, sample
    
    def to(self, device):
        self.device = device
        return self 