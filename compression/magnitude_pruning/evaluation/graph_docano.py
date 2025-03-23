from neuralop.models import CODANO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils import evaluate_model, compare_models
from neuralop.data.transforms.codano_processor import CODANODataProcessor
import matplotlib.pyplot as plt
import numpy as np

fno_model = CODANO(
    in_channels=1,
    output_variable_codimension=1,

    hidden_variable_codimension=2,
    lifting_channels=4,

    use_positional_encoding=False,
    positional_encoding_dim=1,
    positional_encoding_modes=[8, 8],

    use_horizontal_skip_connection=True,
    horizontal_skips_map={3: 1, 4: 0},

    n_layers=5,
    n_heads=[2, 2, 2, 2, 2],
    n_modes=[[8, 8], [8, 8], [8, 8], [8, 8], [8, 8]],
    attention_scaling_factors=[0.5, 0.5, 0.5, 0.5, 0.5],
    per_layer_scaling_factors=[[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]],

    static_channel_dim=0,
    variable_ids=["a1"],
    enable_cls_token=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model.load_model(torch.load("model-codano-darcy-16-resolution-2025-02-11-21-13.pt", weights_only=False))
fno_model.eval()
fno_model = fno_model.to(device)


train_loader, test_loaders, data_processor = load_darcy_flow_small(
    n_train=1000,
    batch_size=16,
    test_resolutions=[16, 32],
    n_tests=[100, 50],
    test_batch_sizes=[16, 16],
    encode_input=False, 
    encode_output=False,
)
test_loader_16 = test_loaders[16]
test_loader_32 = test_loaders[32]

# When creating data processor, use the imported class
data_processor = CODANODataProcessor(
    in_normalizer=data_processor.in_normalizer,
    out_normalizer=data_processor.out_normalizer
)

# Define pruning ratios to test
pruning_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
results_by_ratio = {}

for ratio in pruning_ratios:
    print(f"\nTesting pruning ratio: {ratio:.2%}")
    pruned_model = CompressedModel(
        model=fno_model,
        compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=ratio),
        create_replica=True
    )
    pruned_model = pruned_model.to(device)
    
    results = compare_models(
        model1=fno_model,
        model2=pruned_model,
        test_loaders=test_loaders,
        data_processor=data_processor,
        device=device,
        verbose=False
    )
    results_by_ratio[ratio] = results

# Plotting
plt.figure(figsize=(10, 6))
markers = ['o', 's']  # Different markers for different resolutions
colors = ['#2ecc71', '#e74c3c']  # Different colors for L2 and H1

for resolution in test_loaders.keys():
    # Extract L2 relative errors
    l2_errors = [((results_by_ratio[ratio][f"{resolution}_compressed"]['l2_loss'] / 
                   results_by_ratio[ratio][f"{resolution}_base"]['l2_loss'] - 1) * 100)
                 for ratio in pruning_ratios]
    
    # Extract H1 relative errors
    h1_errors = [((results_by_ratio[ratio][f"{resolution}_compressed"]['h1_loss'] / 
                   results_by_ratio[ratio][f"{resolution}_base"]['h1_loss'] - 1) * 100)
                 for ratio in pruning_ratios]
    
    # Plot lines
    plt.plot(np.array(pruning_ratios) * 100, l2_errors, 
             f'-{markers[0]}', label=f'L2 Loss ({resolution}x{resolution})', 
             color=colors[0], alpha=0.7)
    plt.plot(np.array(pruning_ratios) * 100, h1_errors, 
             f'-{markers[1]}', label=f'H1 Loss ({resolution}x{resolution})', 
             color=colors[1], alpha=0.7)

plt.xlabel('Pruning Ratio (%)')
plt.ylabel('Relative Error Increase (%)')
plt.title('Model Performance vs Magnitude-based Pruning Ratio for Darcy Equation on DOCANO')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('pruning_performance.png', dpi=300, bbox_inches='tight')
plt.show()