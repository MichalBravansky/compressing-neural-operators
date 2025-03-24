from neuralop.models import CODANO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small
from compression.utils.evaluation_util import evaluate_model, compare_models
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
fno_model.load_model(torch.load("models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt", weights_only=False))
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
lowrank_ratios = [0.2,0.4,0.6,0.8]
results_by_ratio = {}

for ratio in lowrank_ratios:
    print(f"\nTesting low rank ratio: {ratio:.2%}")
    lowrank_model = CompressedModel(
        model=fno_model,
        compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio,
                                                       min_rank=8, max_rank=16),
        create_replica=True
    )
    lowrank_model = lowrank_model.to(device)
    
    results = compare_models(
        model1=fno_model,
        model2=lowrank_model,
        test_loaders=test_loaders,
        data_processor=data_processor,
        device=device,
        verbose=False
    )
    results_by_ratio[ratio] = results

# Plotting
metrics = list(next(iter(next(iter(results_by_ratio.values())).values())).keys())
num_metrics = len(metrics)
num_cols = 2
num_rows = (num_metrics + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
axes = axes.flatten()

markers = ['o', 's']
colors = ['#2ecc71', '#e74c3c']

for i, metric in enumerate(metrics):
    for resolution in test_loaders.keys():
        base_values = [results_by_ratio[ratio][f"{resolution}_base"][metric] for ratio in lowrank_ratios]
        compressed_values = [results_by_ratio[ratio][f"{resolution}_compressed"][metric] for ratio in lowrank_ratios]
        
        ax = axes[i]
        ax.plot(np.array(lowrank_ratios) * 100, base_values, 
                f'-{markers[0]}', label=f'Base {metric} ({resolution}x{resolution})', 
                color=colors[0], alpha=0.7)
        ax.plot(np.array(lowrank_ratios) * 100, compressed_values, 
                f'-{markers[1]}', label=f'Compressed {metric} ({resolution}x{resolution})', 
                color=colors[1], alpha=0.7)
        
        ax.set_xlabel('Low Rank Ratio (%)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the spacing between subplots
plt.savefig('compression/LowRank/results/docano_lowrank_performance.png', dpi=300, bbox_inches='tight')
plt.show()