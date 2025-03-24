from compression.quantization.quantized_spectral_layer import *

class SpectralConvObserverCompatible(QuantizedSpectralConv):
    def __init__(self, spectral_layer):
        super().__init__(spectral_layer)
    
    @classmethod
    def from_float(cls, new):
        return SpectralConvObserverCompatible(new)