from neuralop import get_model
from configmypy import ConfigPipeline, YamlConfig
from neuralop.data.datasets import CarCFDDataset
from neuralop.data.transforms.gino_processor import GINOCFDDataProcessor
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.base import CompressedModel
from compression.utils import evaluate_model, compare_models
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
import torch

config_name = 'cfd'
pipe = ConfigPipeline([YamlConfig('./gino_carcfd_config.yaml', config_name=config_name, config_folder='./config')])
config = pipe.read_conf()

gino_model = get_model(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gino_model.load_state_dict(torch.load("model-gino-carcfd-32-resolution-2025-02-12-18-47.pt", weights_only=False))
gino_model.eval()
gino_model = gino_model.to(device)

data_module = CarCFDDataset(
    root_dir=config.data.root, 
    query_res=[config.data.sdf_query_resolution]*3,
    n_train=config.data.n_train,
    n_test=config.data.n_test,
    download=config.data.download
)

test_loader = data_module.test_loader(batch_size=1, shuffle=False)
output_encoder = deepcopy(data_module.normalizers['press']).to(device)
data_processor = GINOCFDDataProcessor(normalizer=output_encoder, device=device)

# Define pruning ratios to test
pruning_ratios = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
results_by_ratio = {}

for ratio in pruning_ratios:
    print(f"\nTesting pruning ratio: {ratio:.2%}")
    pruned_model = CompressedModel(
        model=gino_model,
        compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=ratio),
        create_replica=True
    )
    pruned_model = pruned_model.to(device)
    
    results = compare_models(
        model1=gino_model,
        model2=pruned_model,
        test_loaders={'test': test_loader},
        data_processor=data_processor,
        device=device,
        verbose=False
    )
    results_by_ratio[ratio] = results

plt.figure(figsize=(10, 6))
markers = ['o', 's']
colors = ['#2ecc71', '#e74c3c']

l2_errors = [((results_by_ratio[ratio]['test_compressed']['l2_loss'] / 
               results_by_ratio[ratio]['test_base']['l2_loss'] - 1) * 100)
             for ratio in pruning_ratios]

h1_errors = [((results_by_ratio[ratio]['test_compressed']['h1_loss'] / 
               results_by_ratio[ratio]['test_base']['h1_loss'] - 1) * 100)
             for ratio in pruning_ratios]

plt.plot(np.array(pruning_ratios) * 100, l2_errors, 
         f'-{markers[0]}', label='L2 Loss', 
         color=colors[0], alpha=0.7)
plt.plot(np.array(pruning_ratios) * 100, h1_errors, 
         f'-{markers[1]}', label='H1 Loss', 
         color=colors[1], alpha=0.7)

plt.xlabel('Pruning Ratio (%)')
plt.ylabel('Relative Error Increase (%)')
plt.title('Model Performance vs Magnitude-based Pruning Ratio for Car CFD on GINO')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('gino_pruning_performance.png', dpi=300, bbox_inches='tight')
plt.show()