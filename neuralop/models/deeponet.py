from functools import partialmethod
from typing import Tuple, List, Union, Literal, Optional
from ..layers.normalization_layers import AdaIN, InstanceNorm
Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import LinearChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel
from ..layers.skip_connections import skip_connection
from ..utils import validate_scaling_factor

class DeepONet(BaseModel, name='DeepONet'):
    """Neural Operator that learns a mapping between function spaces using the DeepONet
    architecture as described in [1]_. The DeepONet consists of two networks:
    
    1. A branch network that processes the input function
    2. A trunk network that processes the output coordinates
    
    Parameters
    ----------
    in_channels : int
        Number of channels in input function
    out_channels : int
        Number of channels in output function
    hidden_channels : int
        Width of the networks (i.e. number of channels)
    branch_layers : List[int]
        List of hidden layer sizes for branch network
    trunk_layers : List[int]
        List of hidden layer sizes for trunk network
    positional_embedding : Union[str, nn.Module]
        Positional embedding to apply to coordinates, defaults to "grid"
    non_linearity : nn.Module
        Non-Linear activation function module to use, by default F.gelu
    norm : Literal["ada_in", "group_norm", "instance_norm"]
        Normalization layer to use, by default None
    dropout : float
        Dropout rate to use in MLPs, default 0.0
    
    References
    ----------
    .. [1] Lu Lu, Pengzhan Jin, and George Em Karniadakis. "DeepONet: Learning nonlinear 
           operators for identifying differential equations based on the universal 
           approximation theorem of operators." (2020) https://arxiv.org/abs/1910.03193
    """
    def __init__(
        self,
        train_resolution: int,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        branch_layers: List[int],
        trunk_layers: List[int],
        positional_embedding: Union[str, nn.Module]="grid",
        non_linearity: nn.Module=F.gelu,
        norm: Literal["ada_in", "group_norm", "instance_norm"]=None,
        dropout: float = 0.0,
    ):
        super().__init__()

        branch_layers[-1] = train_resolution * train_resolution * out_channels
        trunk_layers[-1] = train_resolution * train_resolution * out_channels
        self.train_resolution = train_resolution
        self.n_dim = 2  # Assuming 2D for now, can be made flexible
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        
        # Setup separate positional embeddings for branch and trunk
        if positional_embedding == "grid":
            self.branch_pos_embedding = GridEmbeddingND(
                in_channels=in_channels,
                dim=2,
                grid_boundaries=[[0., 1.], [0., 1.]]
            )
            self.trunk_pos_embedding = GridEmbeddingND(
                in_channels=in_channels,
                dim=2,
                grid_boundaries=[[0., 1.], [0., 1.]]
            )
        else:
            self.branch_pos_embedding = None
            self.trunk_pos_embedding = None

        # Branch network
        if self.branch_pos_embedding:
            branch_dims = [(in_channels + 2) * train_resolution * train_resolution] + branch_layers  # Add positional embedding channels to input
        else:
            branch_dims = [in_channels * train_resolution * train_resolution] + branch_layers
            
        self.branch_net = nn.ModuleList()
        for i in range(len(branch_dims)-1):
            self.branch_net.append(
                LinearChannelMLP(
                    layers=[branch_dims[i], branch_dims[i+1]],
                    non_linearity=non_linearity,
                    dropout=dropout
                )
            )

        # Trunk network
        trunk_in_dim = 3 if positional_embedding is not None else 1  # 2 for x,y coordinates + 1 for positional embeddings
        trunk_dims = [trunk_in_dim * train_resolution * train_resolution] + trunk_layers
        self.trunk_net = nn.ModuleList()
        for i in range(len(trunk_dims)-1):
            self.trunk_net.append(
                LinearChannelMLP(
                    layers=[trunk_dims[i], trunk_dims[i+1]],
                    non_linearity=non_linearity,
                    dropout=dropout
                )
            )

 
        self.norm = None

        self.non_linearity = F.gelu

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """DeepONet forward pass
        
        Parameters
        ----------
        x : tensor
            Input function values, shape (batch, channels, height, width)
        y : tensor
            Coordinate points, shape (batch, channels, height, width)
            
        Returns
        -------
        tensor
            Output function values, shape (batch, out_channels, height, width)
        """
        batch_size, _, height, width = x.shape
        
        # Branch network on input function
        branch = x
        if self.branch_pos_embedding is not None:
            branch = self.branch_pos_embedding(branch)

        branch = branch.view(batch_size, -1)

        for i, layer in enumerate(self.branch_net):
            branch = layer(branch)
            if i < len(self.branch_net) - 1:
                branch = self.non_linearity(branch)
        
        # Trunk network on coordinates
        trunk = y
        if self.trunk_pos_embedding is not None:
            trunk = self.trunk_pos_embedding(trunk)

        trunk = trunk.view(batch_size, -1)
        
        for i, layer in enumerate(self.trunk_net):
            trunk = layer(trunk)
            if i < len(self.trunk_net) - 1:
                trunk = self.non_linearity(trunk)
        
        # Reshape branch and trunk to match spatial dimensions
        branch = branch.view(batch_size, -1, height, width)
        trunk = trunk.view(batch_size, -1, height, width)
        
        # Element-wise multiplication and sum across channel dimension
        out = (branch * trunk).sum(dim=1, keepdim=True)
        
        # Add bias
        out = out + self.bias
        
        return out