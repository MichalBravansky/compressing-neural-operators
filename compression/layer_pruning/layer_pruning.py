import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from neuralop.layers.foundation_fno_layers import SpectralConv2dV2
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.models.deeponet import DeepONet  # for type checking

# For models (like FNO) that require an identity preserving transform interface.


class IdentityWithTransform(nn.Module):
    def forward(self, x, **kwargs):
        return x

    def transform(self, x, **kwargs):
        return x

# For DeepONet, we define an alternative: instead of fully removing a layer,
# we want to partially zero-out its weights (using PyTorch pruning).
# We'll use the built-in pruning function.


class GlobalLayerPruning:
    def __init__(self, model):
        self.model = model

    def get_prunable_layers_generic(self):
        """
        Generic candidate search (used for FNO and similar models).
        Candidates:
          - Linear layers (with in_features == out_features)
          - Conv2d layers with 1x1 kernels and equal in/out channels
          - Custom Fourier layers (SpectralConv, SpectralConv2d, SpectralConv2dV2)
        """
        candidate_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                candidate_layers.append((name, module))
            elif isinstance(module, nn.Conv2d) and module.kernel_size == (1, 1) and module.in_channels == module.out_channels:
                candidate_layers.append((name, module))
            elif module.__class__.__name__ in ["SpectralConv", "SpectralConv2d", "SpectralConv2dV2"]:
                candidate_layers.append((name, module))
        return candidate_layers

    def get_prunable_layers_deeponet(self):
        """
        Specialized candidate search for DeepONet.
        Iterates over the branch and trunk ModuleLists and selects Linear layers
        where in_features == out_features.
        """
        candidate_layers = []
        # Iterate over branch network.
        for i, mlp in enumerate(self.model.branch_net):
            for name, module in mlp.named_modules():
                full_name = f"branch_net.{i}"
                if name:
                    full_name += "." + name
                if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                    candidate_layers.append((full_name, module))
        # Iterate over trunk network.
        for i, mlp in enumerate(self.model.trunk_net):
            for name, module in mlp.named_modules():
                full_name = f"trunk_net.{i}"
                if name:
                    full_name += "." + name
                if isinstance(module, nn.Linear) and module.in_features == module.out_features:
                    candidate_layers.append((full_name, module))
        return candidate_layers

    def layer_prune(self, prune_ratio=0.2, deeponet_partial=True):
        """
        Prune candidate layers based on an importance score.

        - For FNO (and similar models), candidates are selected using the generic search and full layer pruning is applied (replacing the layer with an identity).
        - For DeepONet, if deeponet_partial is True, we apply partial (unstructured) pruning to each candidate layer 
          in the branch and trunk networks using torch.nn.utils.prune.l1_unstructured.

          This zeros out a fraction (prune_ratio) of the weights with the smallest magnitudes.

        Parameters:
            prune_ratio (float): For FNO, the fraction of candidate layers to fully prune;
                                 for DeepONet, the fraction of weights to zero out in each candidate layer.
            deeponet_partial (bool): Whether to apply partial pruning for DeepONet.

        Returns:
            List of names of layers that were pruned.
        """
        if isinstance(self.model, DeepONet) and deeponet_partial:
            candidate_layers = self.get_prunable_layers_deeponet()
            if not candidate_layers:
                print("No candidate layers found for DeepONet.")
                return []
            pruned_names = []
            for name, module in candidate_layers:
                # Apply partial pruning using l1_unstructured on the weight.
                prune.l1_unstructured(
                    module, name='weight', amount=prune_ratio)
                # Remove the pruning reparameterization so the pruned weights become permanent.
                prune.remove(module, 'weight')
                pruned_names.append(name)
                print(
                    f"Partially pruned {name} (zeroed out {prune_ratio*100:.1f}% of its weights)")
            return pruned_names
        else:
            # Generic full layer pruning.
            candidate_layers = self.get_prunable_layers_generic()
            if not candidate_layers:
                print("No candidate layers for pruning found.")
                return []
            scores = []
            for name, module in candidate_layers:
                if hasattr(module, 'weights1'):
                    score1 = torch.mean(torch.abs(module.weights1)).item()
                    if hasattr(module, 'weights2'):
                        score2 = torch.mean(torch.abs(module.weights2)).item()
                        score = (score1 + score2) / 2.0
                    else:
                        score = score1
                elif hasattr(module, 'weight') and module.weight is not None:
                    score = torch.mean(torch.abs(module.weight)).item()
                else:
                    score = float('inf')
                scores.append((name, module, score))
            scores.sort(key=lambda x: x[2])
            num_to_prune = int(len(scores) * prune_ratio)
            pruned_layers = scores[:num_to_prune]
            pruned_names = [entry[0] for entry in pruned_layers]
            for name, module, score in pruned_layers:
                parts = name.split('.')
                parent = self.model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                last_part = parts[-1]
                dummy = IdentityWithTransform() if hasattr(
                    module, 'transform') else nn.Identity()
                setattr(parent, last_part, dummy)
            print(f"Pruned layers: {pruned_names}")
            return pruned_names


# Demonstration usage for DeepONet.
if __name__ == "__main__":
    from neuralop.models.deeponet import DeepONet
    import torch.nn.functional as F
    # Instantiate a DeepONet model with example parameters.
    model = DeepONet(
        train_resolution=128,
        in_channels=1,
        out_channels=1,
        hidden_channels=64,
        branch_layers=[256, 256, 256, 256, 128],
        trunk_layers=[256, 256, 256, 256, 128],
        positional_embedding="grid",
        non_linearity=F.gelu,
        norm=None,
        dropout=0.0,
    )

    print("DeepONet model before partial pruning:")
    print(model)

    pruner = GlobalLayerPruning(model)
    pruner.layer_prune(prune_ratio=0.2, deeponet_partial=True)

    # Evaluate the pruned DeepONet with dummy data.
    dummy_x = torch.randn(4, 1, 128, 128)  # Input function
    dummy_y = torch.randn(4, 1, 128, 128)  # Coordinate grid
    output = model(dummy_x, dummy_y)
    print("DeepONet output shape after partial pruning:", output.shape)
