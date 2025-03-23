import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports for spectral layer and quantization wrappers.
# ---------------------------------------------------------------------------
from neuralop.layers.spectral_convolution import SpectralConv
from compression.quantization.quantized_spectral_layer import QuantizedSpectralConv
from compression.quantization.dynamic_quantization import DynamicQuantization

# =============================================================================
# PART 1: Weight Comparison
# =============================================================================
# Create an original spectral layer instance.
orig_layer = SpectralConv(
    in_channels=16,
    out_channels=32,
    n_modes=(16, 16),           # 16 modes per spatial dimension.
    complex_data=False,         # Assume real data.
    max_n_modes=(16, 16),
    bias=True,
    separable=False,
    resolution_scaling_factor=None,
    fno_block_precision="full",
    rank=0.5,
    factorization=None,         # Use Dense weights.
    implementation="reconstructed",
    fixed_rank_modes=False,
    decomposition_kwargs=None,
    init_std="auto",
    fft_norm="forward"
)

# Extract the original weight tensor.
# If the weight has a to_tensor() method (e.g., DenseTensor), use it.
if hasattr(orig_layer.weight, "to_tensor"):
    original_weight = orig_layer.weight.to_tensor().clone()
else:
    original_weight = orig_layer.weight.clone()

# =============================================================================
# PART 2: Output Comparison
# =============================================================================
# Define a test model that contains a spectral layer.
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectral = SpectralConv(
            in_channels=16,
            out_channels=32,
            n_modes=(16, 16),
            complex_data=False,
            max_n_modes=(16, 16),
            bias=True,
            separable=False,
            resolution_scaling_factor=None,
            fno_block_precision="full",
            rank=0.5,
            factorization=None,
            implementation="reconstructed",
            fixed_rank_modes=False,
            decomposition_kwargs=None,
            init_std="auto",
            fft_norm="forward"
        )
    
    def forward(self, x):
        return self.spectral(x)

# Instantiate the test model.
model = TestModel()
# Create a fixed random input tensor.
input_tensor = torch.randn(1, 16, 32, 32)
# Run a forward pass using the original (float) spectral layer.
output_orig = model(input_tensor)






'''
Other compression methods can be added
'''
# =============================================================================
# PART 3: Apply Dynamic Quantization and Extract Dequantized Weights
# =============================================================================
# Apply dynamic quantization to the model.
dq = DynamicQuantization(model)
quantized_model = dq.compress()  # Replaces SpectralConv with QuantizedSpectralConv.

# Run a forward pass using the quantized (dequantized) spectral layer.
output_quant = quantized_model(input_tensor)

# Extract the dequantized weight from the quantized spectral layer.
quant_layer = quantized_model.spectral
if quant_layer.q_weight is not None:
    dequant_weight = quant_layer._dequantize_int8(
        quant_layer.q_weight,
        quant_layer.w_scale,
        quant_layer.w_min.item()
    )
elif quant_layer.q_real is not None and quant_layer.q_imag is not None:
    W_real = quant_layer._dequantize_int8(
        quant_layer.q_real,
        quant_layer.scale_real,
        quant_layer.min_real.item()
    )
    W_imag = quant_layer._dequantize_int8(
        quant_layer.q_imag,
        quant_layer.scale_imag,
        quant_layer.min_imag.item()
    )
    dequant_weight = torch.complex(W_real, W_imag)
else:
    raise RuntimeError("No quantized weight available in the spectral layer.")

# =============================================================================
# PART 4: Compute and Print Comparison Metrics
# =============================================================================
# -- Weight Comparison --
weight_fro_error = torch.norm(original_weight - dequant_weight, p='fro')
weight_rel_error = weight_fro_error / torch.norm(original_weight, p='fro')
weight_spec_error = torch.norm(original_weight - dequant_weight, p=2)
absolute_error1 = torch.abs(original_weight - dequant_weight)

print("Weight Comparison Metrics:")
print("Frobenius Norm of Weight Difference:", weight_fro_error.item())
print("Relative Weight Error:", weight_rel_error.item())
print("Spectral Norm of Weight Difference:", weight_spec_error.item())

# -- Output Comparison --
output_fro_error = torch.norm(output_orig - output_quant, p='fro')
output_rel_error = output_fro_error / torch.norm(output_orig, p='fro')
output_spec_error = torch.norm(output_orig - output_quant, p=2)
absolute_error2 = torch.abs(output_orig - output_quant)

print("\nOutput Comparison Metrics:")
print("Frobenius Norm of Output Difference:", output_fro_error.item())
print("Relative Output Error:", output_rel_error.item())
print("Spectral Norm of Output Difference:", output_spec_error.item())
