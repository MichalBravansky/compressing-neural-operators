from neuralop.models.fno import FNO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from neuralop.data.datasets import load_darcy_flow_small

from compression.utils.evaluation_util import evaluate_model, compare_models, compare_models_hyperparams
from compression.utils.count_params_util import count_selected_layers
from compression.utils.fno_util import optional_fno
#import pandas as pd


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

fno_model, validation_loaders, test_loaders, data_processor = optional_fno(resolution="medium")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fno_model = fno_model.to(device)


# Initialize models 
# pruned_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.05),
#     create_replica=True
# )
# pruned_model = pruned_model.to(device)

# hyperparameters = [0.6]
# models = []
# for ratio in hyperparameters:   
#     lowrank_model = CompressedModel(
#         model=fno_model,
#         compression_technique=lambda model: SVDLowRank(model, 
#                                                     rank_ratio=ratio, # option =  [0.5, 0.55, 0.6, 0.65, 0.68]// [0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
#                                                     min_rank=1,
#                                                     max_rank=256, # option = [8, 16, 32, 64, 128, 256]
#                                                     is_compress_conv1d=False,
#                                                     is_compress_FC=False,
#                                                     is_compress_spectral=True),
#         create_replica=True
#     )
#     lowrank_model = lowrank_model.to(device)
#     models.append(lowrank_model)

# dynamic_quant_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.7, 
#                                                    min_rank=8, max_rank=16),
#     create_replica=True
# )
# lowrank_model = lowrank_model.to(device)

# dynamic_quant_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: DynamicQuantization(model),
#     create_replica=True
# )
# dynamic_quant_model = dynamic_quant_model.to(device)


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

# print("\n"*2)
# print("Low Ranking.....")
# results = compare_models_hyperparams(
#     model1=fno_model,
#     model2s=models,
#     hyperparameters=hyperparameters,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance=True
# )

# print("\n"*2)
# print("Quantizing.....")
# results = compare_models(
#     model1=fno_model,
#     model2=dynamic_quant_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance = True
# )