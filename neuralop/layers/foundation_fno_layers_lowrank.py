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


import torch
import torch.nn as nn
@torch.jit.script
def compl_mul2d_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ioxyt->stboxy", a, b)
    return torch.stack([tmp[0,0,:,:,:,:] - tmp[1,1,:,:,:,:], tmp[1,0,:,:,:,:] + tmp[0,1,:,:,:,:]], dim=-1)

class SpectralConv2dV2_lowrank(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, rank1, rank2, fac_weight1, fac_weight2):
        """
        Initialize a low-rank spectral convolution layer.

        Parameters:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            modes1: Number of Fourier modes (first dimension).
            modes2: Number of Fourier modes (second dimension).
            rank1, rank2: Low-rank decomposition ranks for the two frequency branches, respectively.
        """
        super(SpectralConv2dV2_lowrank, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Fourier modes (corresponding to the first part of the positive frequency region)
        self.modes2 = modes2
        self.rank1 = rank1
        self.rank2 = rank2
        self.scale = 1 / (in_channels * out_channels)
        self.fac_weight1 = fac_weight1
        self.fac_weight2 = fac_weight2
        # Initialize the weight factors for low-rank decomposition.
        # The last dimension of size 2 represents the real and imaginary parts.
        if fac_weight1:
            self.U1 = nn.Parameter(self.scale * torch.rand(in_channels, rank1, modes1, modes2, 2))
            self.V1 = nn.Parameter(self.scale * torch.rand(rank1, out_channels, modes1, modes2, 2))
        else:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        
        if fac_weight2:
            self.U2 = nn.Parameter(self.scale * torch.rand(in_channels, rank2, modes1, modes2, 2))
            self.V2 = nn.Parameter(self.scale * torch.rand(rank2, out_channels, modes1, modes2, 2))
        else:
            self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

        
    def forward(self, x: torch.Tensor):
        """
        Forward propagation:
        1. Apply 2D FFT on input x to obtain its frequency domain representation.
        2. For the corresponding frequency regions, multiply the input by the U factor first,
           then by the V factor, avoiding reconstruction of the full low-rank weight.
        3. Insert the result into the frequency domain tensor and apply inverse FFT to obtain the spatial output.

        Parameters:
            x: Input tensor with shape (batch, in_channels, H, W).

        Returns:
            Output tensor with shape (batch, out_channels, H, W).
        """
        # Get input dimensions and batch size
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype = x.dtype

        # FFT: Convert input to frequency domain
        x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1), norm='ortho')
        x_ft = torch.view_as_real(x_ft)  # Shape: (B, in_channels, H, W//2+1, 2)

        # Initialize output frequency domain tensor
        out_ft = torch.zeros(batchsize, self.out_channels, size_0, size_1 // 2 + 1, 2, device=x.device)
        if self.fac_weight1:
            # ----- Branch 1: Process the first modes1 x modes2 frequency region -----
            # Extract the corresponding frequency region
            x1 = x_ft[:, :, :self.modes1, :self.modes2, :]  # Shape: (B, in_channels, modes1, modes2, 2)
            # Convert to complex form
            x1_complex = torch.view_as_complex(x1)            # Shape: (B, in_channels, modes1, modes2)

            # Convert U1 and V1 to complex tensors
            U1_complex = torch.view_as_complex(self.U1)         # Shape: (in_channels, rank1, modes1, modes2)
            V1_complex = torch.view_as_complex(self.V1)         # Shape: (rank1, out_channels, modes1, modes2)

            # Step 1: Multiply x1 with U1 and sum over the in_channels dimension.
            # Compute: temp1(b, r, m, n) = sum_{c} x1(b, c, m, n) * U1(c, r, m, n)
            temp1 = torch.einsum("bcmn, crmn -> brmn", x1_complex, U1_complex)
            # Step 2: Multiply temp1 with V1 and sum over the rank dimension.
            # Compute: branch1(b, j, m, n) = sum_{r} temp1(b, r, m, n) * V1(r, j, m, n)
            branch1_complex = torch.einsum("brmn, rjmn -> bjmn", temp1, V1_complex)
            # Convert back to real representation, shape: (B, out_channels, modes1, modes2, 2)
            branch1_real = torch.view_as_real(branch1_complex)
            # Assign to the corresponding region in the output frequency tensor
            out_ft[:, :, :self.modes1, :self.modes2, :] = branch1_real
        else:
            out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        if self.fac_weight2:
            # ----- Branch 2: Process the last modes1 x modes2 frequency region -----
            x2 = x_ft[:, :, -self.modes1:, :self.modes2, :]  # Shape: (B, in_channels, modes1, modes2, 2)
            x2_complex = torch.view_as_complex(x2)

            U2_complex = torch.view_as_complex(self.U2)         # Shape: (in_channels, rank2, modes1, modes2)
            V2_complex = torch.view_as_complex(self.V2)         # Shape: (rank2, out_channels, modes1, modes2)

            temp2 = torch.einsum("bcmn, crmn -> brmn", x2_complex, U2_complex)
            branch2_complex = torch.einsum("brmn, rjmn -> bjmn", temp2, V2_complex)
            branch2_real = torch.view_as_real(branch2_complex)
            out_ft[:, :, -self.modes1:, :self.modes2, :] = branch2_real
        else:
            out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d_v2(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
            
        # Convert output frequency tensor to complex (for irfft2)
        out_ft = torch.view_as_complex(out_ft)
        # Apply inverse FFT to obtain spatial domain output
        x_out = torch.fft.irfft2(out_ft, dim=(-2, -1), norm='ortho', s=(size_0, size_1)).to(dtype)
        return x_out


