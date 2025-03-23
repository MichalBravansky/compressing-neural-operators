#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.fft

# Adjust these imports to match your project structure:
from neuralop.layers.spectral_convolution import SpectralConv
from compression.quantization.quantized_spectral_layer import QuantizedSpectralConv

def mse(a, b):
    """Compute Mean Squared Error between two tensors."""
    return torch.mean((a - b) ** 2).item()

def debug_spectral_full():
    print("\n===== FULL DEBUG: SpectralConv (Unquantized vs. Quantized) =====")
    torch.manual_seed(0)  # Fix the seed for reproducibility
    
    # --- Parameters ---
    B = 2
    Cin = 32
    Nx = 16
    Ny = 16
    out_channels = 32
    n_modes = (12, 12)  # e.g., keep 12 modes per spatial dimension
    
    # --- Create a random input ---
    x = torch.randn(B, Cin, Nx, Ny)
    
    # --- Instantiate the unquantized spectral layer ---
    unquant_layer = SpectralConv(
        in_channels=Cin,
        out_channels=out_channels,
        n_modes=n_modes,
        bias=True,
        complex_data=False  # adjust if needed
    )
    
    # --- Instantiate the quantized layer from the unquantized layer ---
    quant_layer = QuantizedSpectralConv(unquant_layer)
    
    # ========================================================
    # (A) Weight Dequantization Error
    # ========================================================
    print("\n--- (A) Weight Comparison (Dequantization Error) ---")
    # If the original weight is factorized, convert it to a raw tensor.
    w_orig_obj = unquant_layer.weight
    if hasattr(w_orig_obj, "to_tensor"):
        w_orig = w_orig_obj.to_tensor()
    else:
        w_orig = w_orig_obj

    with torch.no_grad():
        w_deq = quant_layer._dequantize_weight()
    
    if w_orig.is_complex():
        mse_real = mse(w_orig.real, w_deq.real)
        mse_imag = mse(w_orig.imag, w_deq.imag)
        print(f"Real-part Weight MSE: {mse_real}")
        print(f"Imag-part Weight MSE: {mse_imag}")
    else:
        w_mse = mse(w_orig, w_deq)
        print(f"Weight MSE: {w_mse}")
    
    # ========================================================
    # (B) Bias Dequantization Error
    # ========================================================
    print("\n--- (B) Bias Comparison (Dequantization Error) ---")
    if unquant_layer.bias is not None:
        b_orig = unquant_layer.bias
        b_deq = quant_layer._dequantize_int16(quant_layer.q_bias,
                                               quant_layer.b_scale,
                                               quant_layer.b_min.item())
        mse_bias = mse(b_orig, b_deq)
        print(f"Bias MSE: {mse_bias}")
        print(f"Original Bias Range: min={b_orig.min().item()}, max={b_orig.max().item()}")
        print(f"Dequantized Bias Range: min={b_deq.min().item()}, max={b_deq.max().item()}")
    else:
        print("No bias present in the original layer.")
    
    # ========================================================
    # (C) Intermediate Frequency-Domain Error
    # ========================================================
    print("\n--- (C) Intermediate Frequency-Domain Comparison ---")
    fft_dims = (-2, -1)  # last two dimensions for 2D FFT
    
    with torch.no_grad():
        # Compute FFT of x (using rFFTN since complex_data=False)
        x_fft = torch.fft.rfftn(x, dim=fft_dims, norm=unquant_layer.fft_norm)
        B_, Cin_, Nx_, Ny_r_ = x_fft.shape
        
        # Get weight shapes from original weight
        NxW, NyW = w_orig.shape[-2], w_orig.shape[-1]
        
        # Prepare output FFT buffers for unquantized and quantized paths.
        out_fft_unq = torch.zeros((B_, out_channels, Nx_, Ny_r_), dtype=x_fft.dtype, device=x_fft.device)
        out_fft_q   = torch.zeros_like(out_fft_unq)
        
        # Center-based slicing: For each spatial dim, center the slice around zero frequency.
        center_x = Nx_ // 2
        neg_x = NxW // 2
        pos_x = NxW - neg_x
        slice_x = slice(center_x - neg_x, center_x + pos_x)
        
        center_y = Ny_r_ // 2
        neg_y = NyW // 2
        pos_y = NyW - neg_y
        slice_y = slice(center_y - neg_y, center_y + pos_y)
        
        # For weights, use full slices:
        Wx_slice = slice(0, NxW)
        Wy_slice = slice(0, NyW)
        
        # Loop over channels to perform frequency multiplication (same slicing as in forward())
        for c_in in range(Cin_):
            for c_out in range(out_channels):
                out_fft_unq[:, c_out, slice_x, slice_y] += (
                    x_fft[:, c_in, slice_x, slice_y] * w_orig[c_in, c_out, Wx_slice, Wy_slice]
                )
                out_fft_q[:, c_out, slice_x, slice_y] += (
                    x_fft[:, c_in, slice_x, slice_y] * w_deq[c_in, c_out, Wx_slice, Wy_slice]
                )
        
        freq_mse = mse(out_fft_unq, out_fft_q)
        print(f"Frequency-domain MSE (before iFFT): {freq_mse}")
        
        # iFFT to get spatial outputs (before bias addition)
        x_out_unq = torch.fft.irfftn(out_fft_unq, s=(Nx, Ny), dim=fft_dims, norm=unquant_layer.fft_norm)
        x_out_q   = torch.fft.irfftn(out_fft_q,   s=(Nx, Ny), dim=fft_dims, norm=unquant_layer.fft_norm)
        spatial_mse = mse(x_out_unq, x_out_q)
        print(f"Spatial Domain MSE (after iFFT, before bias): {spatial_mse}")
    
    # ========================================================
    # (D) Full Forward Pass Error & Output Statistics
    # ========================================================
    print("\n--- (D) Full Forward Pass Comparison ---")
    with torch.no_grad():
        unquant_out = unquant_layer(x.clone())
        quant_out = quant_layer(x.clone())
    final_mse = mse(unquant_out, quant_out)
    print(f"Final Output MSE (Full Forward): {final_mse}")
    
    # Compute aggregated statistics (min, max, mean, std) for outputs.
    unq_min, unq_max = unquant_out.min().item(), unquant_out.max().item()
    unq_mean, unq_std = unquant_out.mean().item(), unquant_out.std().item()
    quant_min, quant_max = quant_out.min().item(), quant_out.max().item()
    quant_mean, quant_std = quant_out.mean().item(), quant_out.std().item()
    
    print("\nOutput Statistics:")
    print(f"Unquantized Output: min={unq_min}, max={unq_max}, mean={unq_mean:.6f}, std={unq_std:.6f}")
    print(f"Quantized Output:   min={quant_min}, max={quant_max}, mean={quant_mean:.6f}, std={quant_std:.6f}")

if __name__ == "__main__":
    debug_spectral_full()
