import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure compl_mul2d_v2 is available (import it if defined in another file)
#from your_module import compl_mul2d_v2  # Replace 'your_module' with the actual module name if needed
@torch.jit.script
def compl_mul2d_v2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bixys,ioxyt->stboxy", a, b)
    return torch.stack([tmp[0,0,:,:,:,:] - tmp[1,1,:,:,:,:], tmp[1,0,:,:,:,:] + tmp[0,1,:,:,:,:]], dim=-1)

class QuantizedSpectralConv2dV2(nn.Module):
    """
    A quantized version of SpectralConv2dV2 using int16 quantization.
    
    This layer quantizes the two weight tensors (weights1 and weights2) of the original
    SpectralConv2dV2 layer into int16 using a min–max scaling scheme (65,535 steps with an offset of 32,768).
    During the forward pass, the weights are dequantized on the fly and used for frequency-domain multiplication.
    """
    def __init__(self, spectral_layer):
        super().__init__()
        # Copy metadata from the original layer
        self.in_channels = spectral_layer.in_channels
        self.out_channels = spectral_layer.out_channels
        self.modes1 = spectral_layer.modes1
        self.modes2 = spectral_layer.modes2
        self.fft_norm = 'ortho'  # as used in forward pass
        # Note: SpectralConv2dV2 does not use bias
        
        # Quantize weights1
        (q_w1, scale_w1, min_w1, _, _, _) = self._quantize_tensor_int16(spectral_layer.weights1)
        self.register_buffer('q_weights1', q_w1)
        self.weights1_scale = scale_w1
        self.register_buffer('weights1_min', torch.tensor(min_w1, dtype=torch.float32))
        
        # Quantize weights2
        (q_w2, scale_w2, min_w2, _, _, _) = self._quantize_tensor_int16(spectral_layer.weights2)
        self.register_buffer('q_weights2', q_w2)
        self.weights2_scale = scale_w2
        self.register_buffer('weights2_min', torch.tensor(min_w2, dtype=torch.float32))
    
    def _quantize_tensor_int16(self, tensor: torch.Tensor):
        """
        Quantizes a tensor to int16 using min-max scaling.
        For int16, we use 65,535 steps and an offset of 32,768.
        
        Returns:
          (q_tensor, scale, min_val, None, None, None)
        """
        val_min = tensor.min().item()
        val_max = tensor.max().item()
        #scale = (val_max - val_min) / 65535.0
        scale = (val_max - val_min) / 256.0
        if scale == 0:
            scale = 1.0
        #q_tensor = torch.round((tensor - val_min) / scale - 32768).clamp(-32768, 32767).to(torch.int16)
        q_tensor = torch.round((tensor - val_min) / scale - 127).clamp(-128, 127).to(torch.int8)
        return q_tensor, scale, val_min, None, None, None

    def _dequantize_int16(self, q_int16: torch.Tensor, scale: float, min_val: float) -> torch.Tensor:
        """
        Dequantizes an int16 tensor back to float.
        """
        #return (q_int16.float() + 32768) * scale + min_val
        return (q_int16.float() + 127) * scale + min_val

    def _dequantize_weights(self):
        """
        Dequantizes both weights1 and weights2.
        Returns:
          A tuple (weights1, weights2) with the dequantized floating-point tensors.
        """
        weights1 = self._dequantize_int16(self.q_weights1, self.weights1_scale, self.weights1_min.item())
        weights2 = self._dequantize_int16(self.q_weights2, self.weights2_scale, self.weights2_min.item())
        return weights1, weights2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, height, width)
        size_0 = x.size(-2)
        size_1 = x.size(-1)
        batchsize = x.shape[0]
        dtype = x.dtype

        # Compute the 2D real FFT of the input
        x_ft = torch.fft.rfft2(x.float(), dim=(-2, -1), norm=self.fft_norm)
        x_ft = torch.view_as_real(x_ft)  # Shape: (batch, in_channels, height, width//2+1, 2)

        # Prepare an output tensor in the Fourier domain
        out_ft = torch.zeros(batchsize, self.out_channels, size_0, size_1//2 + 1, 2, device=x.device)
        
        # Dequantize the weights on the fly
        weights1, weights2 = self._dequantize_weights()
        
        # Apply the spectral multiplication using the dequantized weights
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d_v2(x_ft[:, :, :self.modes1, :self.modes2], weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d_v2(x_ft[:, :, -self.modes1:, :self.modes2], weights2)
        
        # Convert the modified Fourier coefficients back to a complex tensor
        out_ft_complex = torch.view_as_complex(out_ft)
        
        # Apply the inverse FFT to return to the spatial domain
        x_out = torch.fft.irfft2(out_ft_complex, dim=(-2, -1), norm=self.fft_norm, s=(size_0, size_1)).to(dtype)
        
        return x_out

    def transform(self, x, output_shape=None):
        # Identity transform; modify if needed.
        return x
