''' From PINO: https://github.com/devzhk/PINO/blob/master/models/basics.py '''
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_act(activation):
    if activation == 'tanh':
        func = F.tanh
    elif activation == 'gelu':
        func = F.gelu
    elif activation == 'relu':
        func = F.relu_
    elif activation == 'elu':
        func = F.elu_
    elif activation == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{activation} is not supported')
    return func


def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    return torch.einsum("bix,iox->box", a, b)


def compl_mul2d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixy,ioxy->boxy", a, b)


def compl_mul3d(a, b):
    return torch.einsum("bixyz,ioxyz->boxyz", a, b)

@torch.jit.script
def compl_mul2d_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ioxyt->stboxy", a, b)
    return torch.stack([tmp[0,0,:,:,:,:] - tmp[1,1,:,:,:,:], tmp[1,0,:,:,:,:] + tmp[0,1,:,:,:,:]], dim=-1)

################################################################
# 2d fourier layer
################################################################

class SpectralConv2dV2(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2dV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        #self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1+1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
    def forward(self, x: torch.Tensor):
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype=x.dtype 

        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x.float(), dim=(-2,-1), norm='ortho')
        x_ft = torch.view_as_real(x_ft)

        out_ft = torch.zeros(batchsize, self.out_channels,  size_0, size_1//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        out_ft = torch.view_as_complex(out_ft)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, dim=(-2,-1), norm='ortho', s=(size_0, size_1)).to(dtype)

        return x