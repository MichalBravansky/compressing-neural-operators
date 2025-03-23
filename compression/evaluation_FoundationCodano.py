'''
from original but only evaluation...
'''
import argparse
import torch
import random
import numpy as np
from neuralop.models.get_models import *
from neuralop.layers.variable_encoding import *
from neuralop.data.datasets.ns_dataset import *
from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from compression.utils.evaluation_util import evaluate_model, compare_models
from neuralop.data.transforms.codano_processor import CODANODataProcessor
from compression.utils.count_params_util import count_selected_layers
from compression.utils.codano_util import missing_variable_testing, CodanoYParams

# we should use scatter to accerlaerate

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="?", default="FSI", type=str)
    parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
    args = parser.parse_args()
    config_file = './config/ssl_ns_elastic.yaml'
    print("Loading config", args.config)
    params = CodanoYParams(config_file, args.config, print_params=True)
    torch.manual_seed(params.random_seed)
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    params.config = args.config
    #stage = StageEnum.PREDICTIVE
    stage = StageEnum.RECONSTRUCTIVE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    variable_encoder = None
    token_expander = None

    encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    if params.use_variable_encoding:
        variable_encoder = get_variable_encoder(params)
        token_expander = TokenExpansion(
            sum([params.equation_dict[i] for i in params.equation_dict.keys()]),
            params.n_encoding_channels, params.n_static_channels,
            params.grid_type == 'uniform'
        )

    codano_model = SSLWrapper(params, encoder, decoder, contrastive, predictor, stage=stage)

    print("Setting the Grid")
    mesh = get_mesh(params)
    input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
    codano_model.set_initial_mesh(input_mesh)

    codano_model.load_state_dict(torch.load(params.model_path), strict=False)
    codano_model = codano_model.cuda().eval()

    variable_encoder.load_encoder(
                    "NS", params.NS_variable_encoder_path)
    
    variable_encoder.load_encoder(
                "ES", params.ES_variable_encoder_path)

    if variable_encoder is not None:
        variable_encoder = variable_encoder.cuda().eval()
    if token_expander is not None:
        token_expander = token_expander.cuda().eval()

    dataset = NsElasticDataset(
        params.data_location,
        equation=list(params.equation_dict.keys()),
        mesh_location=params.input_mesh_location,
        params=params
    )

    validation_dataloader, test_dataloader = dataset.get_validation_test_dataloader(
        params.mu_list, params.dt,
        ntrain=params.get('ntrain'),
        ntest=40,
        batch_size = 1,
        train_test_split = 0.001, # just for test
        sample_per_inlet=params.sample_per_inlet
    )

    #import pickle
    #with open("test_dataloader.pkl", "wb") as f:
    #    pickle.dump(test_dataloader, f)
    # with open("test_dataloader.pkl", "rb") as f:
    #    test_dataloader = pickle.load(f)

    grid_non, grid_uni = get_meshes(params, params.grid_size)
    test_augmenter = None

    codano_evaluation_params = {"variable_encoder": variable_encoder,
                                "token_expander": token_expander,
                                "params": params,
                                "stage": stage,
                                "input_mesh":input_mesh}
    print(codano_model)


    # # Compress Models
    # prune_model = CompressedModel(
    #     model=codano_model,
    #     compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.9),
    #     create_replica=True
    # )

    # prune_model = prune_model.to(device)


    # lowrank_model = CompressedModel(
    #     model=codano_model,
    #     compression_technique=lambda model: SVDLowRank(model, 
    #                                                 rank_ratio=0.8, # option = [0.2, 0.4, 0.6, 0.8]
    #                                                 min_rank=1,
    #                                                 max_rank=256, # option = [8, 16, 32, 64, 128, 256]
    #                                                 is_compress_conv1d=False,
    #                                                 is_compress_FC=False,
    #                                                 is_compress_spectral=True),
    #     create_replica=True
    # )
    # lowrank_model = lowrank_model.to(device)

    # # Start compare models
    # print("\n"*2)
    # print("Pruning.....")
    # compare_models(
    #     model1=codano_model,
    #     model2=prune_model,
    #     test_loaders=test_dataloader,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True,
    #     evaluation_params = codano_evaluation_params
    # )

    # print("\n"*2)
    # print("Low Ranking.....")
    # compare_models(
    #     model1=codano_model,
    #     model2=codano_model,
    #     test_loaders=validation_dataloader,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True,
    #     evaluation_params = codano_evaluation_params
    # )