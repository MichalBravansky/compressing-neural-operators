from neuralop.models import CODANO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test
from compression.utils.evaluation_util import evaluate_model, compare_models
from neuralop.data.transforms.codano_processor import CODANODataProcessor

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
device = torch.device('cpu')
fno_model.load_model(torch.load("models/model-codano-darcy-16-resolution-2025-02-11-21-13.pt", weights_only=False))
fno_model.eval()
fno_model = fno_model.to(device)




validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
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
# lowrank_model = CompressedModel(
#     model=fno_model,
#     compression_technique=lambda model: SVDLowRank(model=model, 
#                                                    rank_ratio=0.6, # option = [0.2, 0.4, 0.6, 0.8]
#                                                    min_rank=1,
#                                                    max_rank=8, # option = [8, 16, 32, 64, 128, 256]
#                                                    is_compress_conv1d=True,
#                                                    is_compress_FC=False,
#                                                    is_compress_spectral=False), # no need to factorize spectral due to small
#     create_replica=True
# )
# lowrank_model = lowrank_model.to(device)

#####################################
# Quant
#####################################
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
# compare_models(
#     model1=fno_model,
#     model2=lowrank_model,
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device,
#     track_performance=True
# )


# print("\n"*2)
# print("Dynamic Quantization.....")
# compare_models(
#     model1=fno_model,               # can remain on CPU or GPU, but if device='cpu', it moves it
#     model2=dynamic_quant_model,     # this is dynamic quant model on CPU
#     test_loaders=test_loaders,
#     data_processor=data_processor,
#     device=device
# )

quantised_model = CompressedModel(
    model=fno_model,
    compression_technique=lambda model: UniformQuantisation(model, num_bits=8),
    create_replica=True
)
print("Quantising.....")
compare_models(
    model1=fno_model,
    model2=quantised_model,
    test_loaders=test_loaders,
    data_processor=data_processor,
    device=device
)