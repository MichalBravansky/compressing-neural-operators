from typing import List, Optional, Tuple, Union

from ..utils import validate_scaling_factor

import torch
from torch import nn

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from tltorch.factorized_tensors.core import FactorizedTensor


from typing import List, Optional, Tuple, Union
from copy import deepcopy
from ..utils import validate_scaling_factor
import torch
from torch import nn
import tensorly as tl
from tltorch.factorized_tensors.core import FactorizedTensor
from .base_spectral_conv import BaseSpectralConv

from .einsum_utils import einsum_complexhalf
from .base_spectral_conv import BaseSpectralConv
from .resample import resample

tl.set_backend("pytorch")
use_opt_einsum("optimal")
einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

Number = Union[int, float]

class DoubleSpectralConv(BaseSpectralConv):
    """ Double Sepctral Layer"""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        complex_data=False,
        max_n_modes=None,
        bias=True,
        separable=False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="forward",
        device=None,
        mid_channels=None,  # equal to rank
    ):
        super().__init__(device=device)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.complex_data = complex_data
        self.n_modes = n_modes
        self.order = len(self.n_modes)
        self.max_n_modes = max_n_modes or self.n_modes
        self.fno_block_precision = fno_block_precision
        self.separable = separable
        self.rank = rank
        self.factorization = factorization
        self.implementation = implementation
        self.resolution_scaling_factor = validate_scaling_factor(resolution_scaling_factor, self.order)
        self.fft_norm = fft_norm

        if init_std == "auto":
            init_std1 = (2 / (in_channels + self.mid_channels))**0.5
            init_std2 = (2 / (self.mid_channels + out_channels))**0.5
        else:
            init_std1 = init_std2 = init_std

        self.weight1 = self._create_weight(
            in_channels, self.mid_channels, 
            init_std1, decomposition_kwargs or {}
        )
        
        self.weight2 = self._create_weight(
            self.mid_channels, out_channels,
            init_std2, deepcopy(decomposition_kwargs) or {}
        )

        if bias:
            self.bias = nn.Parameter(
                init_std2 * torch.randn(*(tuple([out_channels]) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _create_weight(self, in_ch, out_ch, init_std, decomp_kwargs):
        # create 2 weights
        if self.separable:
            weight_shape = (in_ch, *self.max_n_modes)
        else:
            weight_shape = (in_ch, out_ch, *self.max_n_modes)
        if self.factorization is None:
            weight = torch.empty(weight_shape, dtype=torch.cfloat, device=self.device)
        weight.normal_(0, init_std)
        return weight

    
    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        
        # from original
        batchsize, channels, *mode_sizes = x.shape
        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1
        fft_dims = list(range(-self.order, 0))

        if self.complex_data:
            x_ft = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
        else:
            x_ft = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.order > 1:
            x_ft = torch.fft.fftshift(x_ft, dim=fft_dims[:-1])

        # key things
        out_ft = self._apply_weight(x_ft, self.weight1, self.weight2, fft_size)

        # from original
        if self.order > 1:
            out_ft = torch.fft.ifftshift(out_ft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(out_ft, s=output_shape, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(out_ft, s=output_shape, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x += self.bias

        return x

    def _apply_weight(self, x_ft, weight1, weight2, fft_size):
        slices_w1, slices_in = self._get_slices(fft_size, weight1.shape)
        slices_w2, slices_out = self._get_slices(fft_size, weight2.shape)

        intermediate = torch.einsum('bihw,irhw->brhw', x_ft[slices_in], weight1[slices_w1])
        result = torch.einsum('brhw,rjhw->bjhw', intermediate, weight2[slices_w2])
        B = x_ft.shape[0]
        out_ft = torch.zeros(B, self.out_channels, *fft_size, device=x_ft.device, dtype=x_ft.dtype)
        out_ft[slices_out] = result
        return out_ft

    # from original spectral
    def _get_slices(self, fft_size, weight_shape):
        starts = [max_size - min(size, n_mode) 
                for max_size, size, n_mode in zip(self.max_n_modes, fft_size, self.n_modes)]
        
        if self.separable:
            slices_w = [slice(None)]
        else:
            slices_w = [slice(None), slice(None)]
        
        slices_w += self._get_spatial_slices(starts)
        slices_x = [slice(None), slice(None)]
        for size, n_mode in zip(fft_size, self.n_modes):
            center = size // 2
            neg = n_mode // 2
            pos = n_mode // 2 + n_mode % 2
            slices_x.append(slice(center-neg, center+pos))
        
        return slices_w, slices_x
    # from original spectral
    def _get_spatial_slices(self, starts):
        slices = []
        for i, start in enumerate(starts):
            if i == len(starts)-1 and not self.complex_data:
                slices.append(slice(None, -start) if start else slice(None))
            else:
                slices.append(slice(start//2, -start//2) if start else slice(None))
        return slices

