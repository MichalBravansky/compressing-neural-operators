from neuralop.models import DeepONet
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test
from compression.utils.evaluation_util import evaluate_model, compare_models
from compression.utils.count_params_util import count_selected_layers
from compression.quantization.dynamic_quantization import DynamicQuantization

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

deeponet_model = DeepONet(
    train_resolution=128,
    in_channels=1,
    out_channels=1, 
    hidden_channels=64,
    branch_layers=[256, 256, 256, 256, 128],
    trunk_layers=[256, 256, 256, 256, 128],
    positional_embedding='grid',
    non_linearity='gelu',
    norm='instance_norm',
    dropout=0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deeponet_model.load_state_dict(torch.load("models/model-deeponet-darcy-128-resolution-2025-03-04-18-53.pt", weights_only=False))
deeponet_model.eval()
deeponet_model = deeponet_model.to(device)

validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
    n_train=10000,
    batch_size=16,
    test_resolutions=[128],
    n_tests=[1000],
    test_batch_sizes=[4, 4],
    encode_input=False, 
    encode_output=False,
)
param_stats = count_selected_layers(deeponet_model)
print(deeponet_model)
for layer_type, count in param_stats.items():
    print(f"{layer_type}: {count} parameters")

'''
pruned_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.5),
    create_replica=True
)
pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: SVDLowRank(model, 
                                                   rank_ratio=0.5, # option = [0.2, 0.4, 0.6, 0.8]
                                                   min_rank=128, # for deeponet, its important to set min_rank to be higher
                                                   max_rank=256, # option = [8, 16, 32, 64, 128, 256]
                                                   is_full_rank=False,
                                                   is_compress_conv1d=False,
                                                   is_compress_FC=True,
                                                   is_comrpess_spectral=False),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)
'''
dynamic_quant_model = CompressedModel(
    model=deeponet_model,
    compression_technique=lambda model: DynamicQuantization(model),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)


'''
print("\n"*2)
print("Pruning.....")
compare_models(
    model1=deeponet_model,
    model2=pruned_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)

print("\n"*2)
print("Low Ranking.....")
compare_models(
    model1=deeponet_model,
    model2=lowrank_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device,
    track_performance = True
)
'''

# Evaluate both models on CPU
print("\n"*2)
print("Dynamic Quantization.....")
compare_models(
    model1=deeponet_model,               # The original model (it will be moved to CPU in evaluate_model)
    model2=dynamic_quant_model,     # The dynamically quantized model
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)
