# Layer Pruning Module

This module implements layer pruning for neural operators. The goal is to remove (or bypass) entire layers based on an importance metric computed from their weights.

## Files

- **layer_pruning.py**: Contains the `GlobalLayerPruning` class that implements the layer pruning logic.
- **evaluation/eval_layer_pruning.py**: A demonstration script that applies layer pruning to a simple model and compares outputs before and after pruning.
- **__init__.py**: Allows the module to be imported as a Python package.

## Usage

To prune a model:
```python
from compression.layer_pruning import GlobalLayerPruning

# Assuming `model` is your neural operator instance.
pruner = GlobalLayerPruning(model)
pruner.layer_prune(prune_ratio=0.2)  # Prune the 20% least important layers.
