import torch.nn as nn
from collections import defaultdict

def count_selected_layers(model):
    counts = defaultdict(int)

    spectral_layer_names = {'SpectralConv', 'SpectralConv2dV2', 'SpectralConvKernel2d'}

    for module in model.modules():
        name = module.__class__.__name__

        # ✅ Spectral Layer
        if name in spectral_layer_names:
            if hasattr(module, 'weight'):
                weight = module.weight

                # Case 1: TuckerTensor-like
                if hasattr(weight, 'core') and hasattr(weight, 'factors'):
                    counts['SpectralConv'] += weight.core.numel()
                    for f in weight.factors:
                        counts['SpectralConv'] += f.numel()

                # Case 2: ModuleList or list of ComplexDenseTensor
                elif isinstance(weight, (nn.ModuleList, list)):
                    for w in weight:
                        if isinstance(w, nn.Parameter):
                            counts['SpectralConv'] += w.numel()
                        elif hasattr(w, 'parameters'):
                            for p in w.parameters():
                                counts['SpectralConv'] += p.numel()
                        elif hasattr(w, 'numel'):  # raw tensor
                            counts['SpectralConv'] += w.numel()

                # Case 3: Just a Parameter
                elif isinstance(weight, nn.Parameter):
                    counts['SpectralConv'] += weight.numel()

                # Case 4: fallback
                else:
                    try:
                        counts['SpectralConv'] += weight.numel()
                    except:
                        pass

            else:
                counts['SpectralConv'] += sum(p.numel() for p in module.parameters(recurse=False))

        # ✅ Conv1d
        elif isinstance(module, nn.Conv1d):
            counts['Conv1d'] += sum(p.numel() for p in module.parameters(recurse=False))

        # ✅ Linear
        elif isinstance(module, nn.Linear):
            counts['Linear'] += sum(p.numel() for p in module.parameters(recurse=False))

    return counts
