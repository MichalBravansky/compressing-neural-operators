#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.fft

# Adjust the import paths to match your project structure:
from neuralop.layers.spectral_convolution import SpectralConv
from compression.quantization.quantized_spectral_layer import QuantizedSpectralConv

def mse(a, b):
    """Computes the Mean Squared Error between two tensors."""
    return torch.mean((a - b) ** 2).item()

def debug_spectral_weights():
    """
    1) Compare weights before & after quantization (including complex parts).
    2) Compare bias before & after quantization.
    3) Run a partial forward pass to check if the error accumulates from bias addition.
    """
    print("\n===== DEBUGGING SPECTRAL WEIGHTS AND BIAS HANDLING =====")

    # ---- Step 1: Create an example SpectralConv layer ----
    in_channels, out_channels, n_modes = 32, 32, (12, 12)
    spectral_layer = SpectralConv(in_channels, out_channels, n_modes)
    
    # ---- Step 2: Quantize it ----
    quantized_layer = QuantizedSpectralConv(spectral_layer)

    # ---- Step 3: Extract weights & biases ----
    # If the original weight is factorized, convert it to a raw tensor.
    w_orig_obj = spectral_layer.weight
    if hasattr(w_orig_obj, "to_tensor"):
        w_orig = w_orig_obj.to_tensor()
    else:
        w_orig = w_orig_obj

    with torch.no_grad():
        # Dequantize the weights from the quantized layer.
        w_deq = quantized_layer._dequantize_weight()

    # ---- Step 4: Compare Weight MSE ----
    print("\n=== Weight Comparison (Complex) ===")
    if w_orig.is_complex():
        mse_real = mse(w_orig.real, w_deq.real)
        mse_imag = mse(w_orig.imag, w_deq.imag)
        print(f"Real-part MSE: {mse_real}")
        print(f"Imag-part MSE: {mse_imag}")
    else:
        w_mse = mse(w_orig, w_deq)
        print(f"Weight MSE: {w_mse}")

    # ---- Step 5: Compare Bias MSE ----
    print("\n=== Bias Comparison ===")
    if spectral_layer.bias is not None:
        b_orig = spectral_layer.bias
        # For int16 quantization, use _dequantize_int16 if available:
        if hasattr(quantized_layer, "_dequantize_int16"):
            b_deq = quantized_layer._dequantize_int16(quantized_layer.q_bias,
                                                       quantized_layer.b_scale,
                                                       quantized_layer.b_min.item())
        else:
            b_deq = quantized_layer._dequantize(quantized_layer.q_bias,
                                                quantized_layer.b_scale,
                                                quantized_layer.b_min.item())

        mse_bias = mse(b_orig, b_deq)
        print(f"Bias MSE: {mse_bias}")
        print(f"Original Bias Range: min={b_orig.min().item()}, max={b_orig.max().item()}")
        print(f"Dequantized Bias Range: min={b_deq.min().item()}, max={b_deq.max().item()}")
    else:
        print("No bias present in the original layer.")

    # ---- Step 6: Run a partial forward pass ----
    print("\n=== Partial Forward Pass Comparison ===")
    x = torch.randn(16, in_channels, 16, 16)  # Example input
    unquant_out = spectral_layer(x)
    quant_out   = quantized_layer(x)
    mse_final = mse(unquant_out, quant_out)
    print(f"Final Output (Full Forward) MSE = {mse_final}")

if __name__ == "__main__":
    debug_spectral_weights()
