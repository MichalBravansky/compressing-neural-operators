'''
this verify fiel is only for the quantization algorithm for the scale/ int8, which is replaced by min-max algorithm, 
so this verifying file is no longer used
'''



#################### test_dynamic_quant.py ####################
import torch
import torch.nn as nn

# Adjust the import below to match the name of your own file/module
# where DynamicQuantization (and possibly QuantizedLinear, QuantizedConv1d) is defined.
from compression.quantization.dynamic_quantization import DynamicQuantization

def test_linear():
    """Test quantization of a simple Linear layer"""
    print("\n=================== LINEAR LAYER TEST ===================")
    
    # 1) Create a simple Linear model
    float_model = nn.Sequential(nn.Linear(in_features=4, out_features=1, bias=False))
    
    # 2) Manually set its weights to [1.11, 2.22, 3.33, 4.44]
    float_model[0].weight.data = torch.tensor(
        [[1.11, 2.22, 3.33, 4.44]], dtype=torch.float32
    )
    
    print("=== BEFORE COMPRESSION ===")
    print("Original float weight:", float_model[0].weight.data)
    
    # Quick forward pass in float mode
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    y_float = float_model(x)
    print("Float model output:", y_float)

    # 3) Apply dynamic quantization
    quantizer = DynamicQuantization(float_model)
    quantized_model = quantizer.compress()
    
    print("\n=== AFTER COMPRESSION ===")
    wrapper_layer = quantized_model[0]  # The wrapped QuantizedLinear
    
    print("Scale used for weight:", wrapper_layer.scale)
    print("Quantized weight (int8):", wrapper_layer.q_weight)
    
    dequantized_w = wrapper_layer.q_weight.float() * wrapper_layer.scale
    print("Dequantized weight:", dequantized_w)
    
    # 5) Forward pass with quantized model
    y_quant = quantized_model(x)
    print("Quantized model output:", y_quant)
    print("========================================================")


def test_conv1d():
    """Test quantization of a simple 1x1 Conv1d layer"""
    print("\n=================== CONV1D LAYER TEST ===================")

    # 1) Create a simple Conv1d model (1x1 kernel size)
    float_model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False))

    # 2) Manually set its weights
    float_model[0].weight.data = torch.tensor([[[1.11, 2.22, 3.33, 4.44]]], dtype=torch.float32)  # Shape [out, in, kernel]

    print("=== BEFORE COMPRESSION ===")
    print("Original float weight:", float_model[0].weight.data)
    
    # Quick forward pass in float mode
    x = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)  # Shape [batch, channels, width]
    y_float = float_model(x)
    print("Float model output:", y_float)

    # 3) Apply dynamic quantization
    quantizer = DynamicQuantization(float_model)
    quantized_model = quantizer.compress()
    
    print("\n=== AFTER COMPRESSION ===")
    wrapper_layer = quantized_model[0]  # The wrapped QuantizedConv1d
    
    print("Scale used for weight:", wrapper_layer.scale)
    print("Quantized weight (int8):", wrapper_layer.q_weight)
    
    dequantized_w = wrapper_layer.q_weight.float() * wrapper_layer.scale
    print("Dequantized weight:", dequantized_w)
    
    # 5) Forward pass with quantized model
    y_quant = quantized_model(x)
    print("Quantized model output:", y_quant)
    print("========================================================")

def test_spectral():
    """
    Test quantization of a minimal Spectral-like layer with a single complex weight.
    We'll define a small class 'SimpleSpectralModel' that:
    1) Takes an input of shape [batch, channels, height, width].
    2) Does an rFFT2 -> multiply by a small complex weight -> iFFT2.
    We'll set that complex weight by hand and see how quantization changes the result.
    """
    print("\n=================== SPECTRAL LAYER TEST ===================")

    import torch
    import torch.nn as nn
    from torch.fft import rfftn, irfftn

    # 1) Define a minimal spectral model with a single complex weight
    class SimpleSpectralConv(nn.Module):
        def __init__(self):
            super().__init__()
            # Let's say we have in_channels=1, out_channels=1, n_modes=(2,2) for demonstration
            self.in_channels = 1
            self.out_channels = 1
            self.n_modes = (2, 2)  # pretend we keep 2 freq modes in each dimension

            # We'll store a small complex64 weight of shape [inC, outC, NxModes, NyModes]
            # e.g. shape [1,1,2,2]
            weight_data = torch.zeros((1, 1, 2, 2), dtype=torch.complex64)
            # Set some example complex entries:
            weight_data[0, 0, 0, 0] = 5 + 1j
            weight_data[0, 0, 0, 1] = 2 + 3j
            weight_data[0, 0, 1, 0] = -1 - 4j
            weight_data[0, 0, 1, 1] = 6 + 7j

            # We'll store this as a Parameter to mimic a real SpectralConv
            self.weight = nn.Parameter(weight_data)

        def forward(self, x):
            """
            1) rFFT2
            2) Multiply by self.weight (shape [1,1,2,2])
            3) iFFT2
            """
            b, c, h, w = x.shape
            # 1) rFFT2 => shape ~ [b, c, h, w//2+1] for real data
            x_fft = rfftn(x, dim=(-2, -1), norm="ortho")

            # For simplicity, assume h>=2, w>=2, 
            # and we just multiply the top-left 2x2 frequencies:
            out_fft = x_fft.clone() * 0  # same shape
            # out_fft shape: [b, c, h, w//2 + 1]
            
            # We'll keep a small slice:
            slice_h = slice(0, self.n_modes[0])  # 0..2
            slice_w = slice(0, self.n_modes[1])  # 0..2

            # Multiply inC->outC => we only have 1->1, so no channel loops needed
            # for a more advanced scenario, we'd sum over c_in -> c_out
            out_fft[:, 0, slice_h, slice_w] = (
                x_fft[:, 0, slice_h, slice_w] * self.weight[0, 0, :, :]
            )

            # 3) iFFT2 => back to real space
            x_out = irfftn(out_fft, s=(h, w), dim=(-2, -1), norm="ortho")
            return x_out

    # 2) Create the model and set the complex weight
    float_model = nn.Sequential(SimpleSpectralModel())

    # Show the original float complex weight
    spectral_layer = float_model[0]
    print("=== BEFORE COMPRESSION ===")
    print("Original complex weight (4 entries):")
    print(spectral_layer.weight)

    # 3) Do a quick forward pass with real data
    # shape [batch=1, channels=1, height=4, width=4]
    x = torch.ones((1, 1, 4, 4), dtype=torch.float32)
    y_float = float_model(x)
    print("Float model output shape:", y_float.shape)
    print("Some example output values (float):", y_float.flatten()[:4])

    # 4) Apply dynamic quantization
    from compression.quantization.dynamic_quantization import DynamicQuantization
    quantizer = DynamicQuantization(float_model)
    quantized_model = quantizer.compress()

    print("\n=== AFTER COMPRESSION ===")
    # Our spectral layer is now replaced with a 'QuantizedSpectralConv' (assuming the code catches it).
    # If your dynamic_quant code is looking for "SpectralConv" in the name, 
    # you might need to rename "SimpleSpectralModel" or forcibly detect it.
    # For demonstration, let's say it replaced the module in the model:
    wrapper_layer = quantized_model[0]
    
    print("\n[Inspecting the compressed spectral layer...]")
    # We might not have direct variables like "core" or "factors" if it's not factorized,
    # or if your code is skipping. But let's do a forward pass again.
    y_quant = quantized_model(x)
    print("Quantized model output shape:", y_quant.shape)
    print("Some example output values (quantized):", y_quant.flatten()[:4])

    # We can compare the difference:
    diff = (y_quant - y_float).abs().mean().item()
    print(f"Mean absolute difference between float & quant outputs: {diff:.6f}")
    print("========================================================")
    print("Type of wrapper_layer:", type(wrapper_layer))
    # Possibly <class 'torch.nn.modules.container.Sequential'> or something

    print("Does wrapper_layer have child modules?", list(wrapper_layer.named_children()))
    # You might see that the actual QSpectral is wrapper_layer[0]


if __name__ == "__main__":
    test_linear()
    test_conv1d()
    #test_spectral() 
