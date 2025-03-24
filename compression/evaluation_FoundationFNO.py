from neuralop.models.foundation_fno import fno
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from compression.utils.fno_util import FNOYParams
from neuralop.data.datasets.mixed import get_data_val_test_loader
from compression.utils.evaluation_util import evaluate_model, compare_models
from compression.utils.count_params_util import count_selected_layers

import os, sys, time
import argparse
import wandb
import matplotlib.pyplot as plt
import logging
import torch.distributed as dist

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--yaml_config", default='config/mixed_config.yaml', type=str)
parser.add_argument("--config", default='fno-foundational', type=str)
parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
parser.add_argument("--run_num", default='0', type=str, help='sub run config')
parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
args = parser.parse_args()
params = FNOYParams(os.path.abspath(args.yaml_config), args.config, print_params=False)
if dist.is_initialized():
    dist.barrier()
logging.info('DONE')
params['global_batch_size'] = params.batch_size
params['local_batch_size'] = int(params.batch_size)

params['global_valid_batch_size'] = params.valid_batch_size
params['local_valid_batch_size'] = int(params.valid_batch_size)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cpu_device = torch.device('cpu')


fno_model = fno(params).to(device)

checkpoints = torch.load("models/ckpt_best.tar", map_location='cpu', weights_only=False)  
fno_model.load_state_dict(checkpoints['model_state'])
fno_model.eval()

validation_dataloaders, test_loaders, data_processor = get_data_val_test_loader(params,
                                                                  params.test_path, 
                                                                  dist.is_initialized(), 
                                                                  train=False, 
                                                                  pack=params.pack_data)

param_stats = count_selected_layers(fno_model)
for layer_type, count in param_stats.items():
    print(f"{layer_type}: {count} parameters")

# print("\n"*2)
# print("Compressing Model.....")
# Initialize models 
# pruned_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.5),
#     create_replica=True
# )
# pruned_model = pruned_model.to(device)

# lowrank_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model, 
#                                                    rank_ratio=0.9, # [0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99]
#                                                    min_rank=1,
#                                                    max_rank=256,
#                                                    is_full_rank=False,
#                                                    is_compress_conv1d=False,
#                                                    is_compress_FC=False,
#                                                    is_compress_spectral=True),
#     create_replica=True
# )

# lowrank_model = lowrank_model.to(device)
# lowrank_model = lowrank_model.to(device)

# dynamic_quant_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model, 
#                                                    rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
#                                                    min_rank=16,
#                                                    max_rank=256, # option = [8, 16, 32, 64, 128, 256]
#                                                    is_full_rank=True,
#                                                    is_compress_conv1d=False,
#                                                    is_compress_FC=False,
#                                                    is_comrpess_spectral=True),
#     create_replica=True
# )

# lowrank_model = lowrank_model.to(device)

dynamic_quant_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: UniformQuantisation(model , num_bits=8, num_calibration_runs=1),
    create_replica=True
)
dynamic_quant_model = dynamic_quant_model.to(device)


# print("\n"*2)
# print("Getting Result.....")
# print("\n"*2)
# print("Pruning.....")
# compare_models(
#     model1=fno_model,
#     model2=pruned_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device
# )


# results = compare_models(
#     model1=fno_model,
#     model2=lowrank_model,
#     test_loaders=validation_dataloaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance = True
# )

# print("\n"*2)
# print("Dynamic Quantization.....")
# compare_models(
#     model1=fno_model,               # The original model (it will be moved to CPU in evaluate_model)
#     model2=dynamic_quant_model,     # The dynamically quantized model
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance = True
# )

# print("\n"*2)
# print("Dynamic Quantization.....")
# compare_models(
#     model1=fno_model,               # The original model (it will be moved to CPU in evaluate_model)
#     model2=dynamic_quant_model,     # The dynamically quantized model
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device
# )

#print(results)