from neuralop.models import CODANO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test
from compression.utils.evaluation_util import evaluate_model, compare_models, compare_models_hyperparams
from compression.utils.count_params_util import count_selected_layers
from neuralop.data.transforms.codano_processor import CODANODataProcessor

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

fno_model = CODANO(
    in_channels=1,
    output_variable_codimension=1,

    hidden_variable_codimension=2,
    lifting_channels=4,

    use_positional_encoding=True,
    positional_encoding_dim=2,
    positional_encoding_modes=[8, 8],

    use_horizontal_skip_connection=True,
    horizontal_skips_map={3: 1, 4: 0},

    n_layers=5,
    n_heads=[32, 32, 32, 32, 32],
    n_modes= [[128, 128], [128, 128], [128, 128], [128, 128], [128, 128]],
    attention_scaling_factors=[0.5, 0.5, 0.5, 0.5, 0.5],
    per_layer_scaling_factors=[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]],

    static_channel_dim=0,
    variable_ids=["a1"],
    enable_cls_token=False
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')
fno_model.load_model(torch.load("models/model-codano-darcy-16-resolution-2025-03-15-19-31.pt", weights_only=False))
fno_model.eval()
fno_model = fno_model.to(device)

validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
    n_train=10000,
    batch_size=16,
    test_resolutions=[128],
    n_tests=[1000],
    test_batch_sizes=[4, 4],
    encode_input=False, 
    encode_output=False,
)

param_stats = count_selected_layers(fno_model)
print(fno_model)
for layer_type, count in param_stats.items():
    print(f"{layer_type}: {count} parameters")

# # When creating data processor, use the imported class
# data_processor = CODANODataProcessor(
#     in_normalizer=data_processor.in_normalizer,
#     out_normalizer=data_processor.out_normalizer
# )

###################################################
# Global Magnitude Pruning  
###################################################
# pruned_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.05),
#     create_replica=True
# )
# pruned_model = pruned_model.to(device)

#####################################
# SVD Low-Rank Decomposition 
#####################################
# there are only few layers can be factorized but after factorization, the loss increased a lot
# low rank is not working on DOCANO

# hyperparameters = {
#     "FNO 16x16": [0.45, 0.5, 0.55, 0.58, 0.6], # small
#     "FNO 32x32": [0.4,0.45,0.47, 0.5, 0.55],
#     "Codano":  [0.5],
#     "FNO 128x128": [0.45, 0.5, 0.51, 0.52, 0.53],
#     "DeepONet": [0.95, 0.96, 0.97, 0.98, 0.99],
# }
# codanos = []
# codano_hyperparams = hyperparameters["Codano"]
# for ratio in codano_hyperparams:
#     codanolowrank_model = CompressedModel(
#         model=fno_model,
#         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio,                                                           
#                                                         is_compress_conv1d=True,
#                                                         is_compress_spectral=False),
#         create_replica=True
#     )
#     codanolowrank_model = codanolowrank_model.to(device)
#     codanos.append(codanolowrank_model)

# codano_compare = compare_models_hyperparams(
#     model1=fno_model,
#     model2s=codanos,
#     hyperparameters=codano_hyperparams,
#     test_loaders=validation_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance=True
# )

#####################################
# Quant
#####################################
dynamic_quant_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: DynamicQuantization(model),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)



# Start Compression ..

# print("\n"*2)
# print("Pruning.....")
# compare_models(
#     model1=fno_model,
#     model2=pruned_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device
# )

print("\n"*2)
print("Dynamic Quantization.....")
compare_models(
    model1=fno_model,               # can remain on CPU or GPU, but if device='cpu', it moves it
    model2=dynamic_quant_model,     # this is dynamic quant model on CPU
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)