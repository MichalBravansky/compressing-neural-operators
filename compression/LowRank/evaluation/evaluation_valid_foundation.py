from neuralop.models import FNO
import torch
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test
import matplotlib.pyplot as plt
import random
import numpy as np
from neuralop.models.get_models import *
from neuralop.layers.variable_encoding import *
from neuralop.data.datasets.ns_dataset import *
from neuralop.data_utils.data_utils import MaskerNonuniformMesh, get_meshes
from compression.utils.codano_util import missing_variable_testing, CodanoYParams
from neuralop.models.foundation_fno import fno
from compression.LowRank.SVD_LowRank import SVDLowRank

from compression.base import CompressedModel
from compression.utils.fno_util import FNOYParams
from neuralop.data.datasets.mixed import get_data_val_test_loader
import argparse
import os
import logging
import torch.distributed as dist
from neuralop.models import CODANO
from neuralop.data.transforms.codano_processor import CODANODataProcessor
from neuralop.models import DeepONet

from compression.utils.evaluation_util import evaluate_model, compare_models, compare_models_hyperparams
from compression.utils.fno_util import optional_fno
from compression.utils.graph_util import generate_graph
import pandas as pd
import pickle

if __name__ == "__main__":

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------------- INIT FOUNDATION FNO MODEL ---------------------------------------
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
    ffno_model = fno(params).to(device)
    checkpoints = torch.load("models/ckpt_best.tar", map_location='cpu', weights_only=False)  
    ffno_model.load_state_dict(checkpoints['model_state'])
    ffno_model.eval()

    validation_loaders_ffno, test_loaders_ffno, data_processor_ffno = get_data_val_test_loader(params,
                                                                    params.test_path, 
                                                                    dist.is_initialized(), 
                                                                    train=False, 
                                                                    pack=params.pack_data)

    # ------------------------------------- INIT FOUNDATION CODANO MODEL ---------------------------------------
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
    stage = StageEnum.RECONSTRUCTIVE
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

    fcodano_model = SSLWrapper(params, encoder, decoder, contrastive, predictor, stage=stage)

    print("Setting the Grid")
    mesh = get_mesh(params)
    input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
    fcodano_model.set_initial_mesh(input_mesh)

    fcodano_model.load_state_dict(torch.load(params.model_path), strict=False)
    fcodano_model = fcodano_model.cuda().eval()

    variable_encoder.load_encoder(
                    "NS", params.NS_variable_encoder_path)

    variable_encoder.load_encoder(
                "ES", params.ES_variable_encoder_path)

    if variable_encoder is not None:
        variable_encoder = variable_encoder.cuda().eval()
    if token_expander is not None:
        token_expander = token_expander.cuda().eval()

    dataset_fcodano = NsElasticDataset(
        params.data_location,
        equation=list(params.equation_dict.keys()),
        mesh_location=params.input_mesh_location,
        params=params
    )

    validation_loaders_fcodano, test_loaders_fcodano = dataset_fcodano.get_validation_test_dataloader(
        params.mu_list, params.dt,
        ntrain=params.get('ntrain'),
        ntest=40,
        batch_size = 1,
        sample_per_inlet=params.sample_per_inlet
    )

    grid_non, grid_uni = get_meshes(params, params.grid_size)
    test_augmenter = None

    fcodano_evaluation_params = {"variable_encoder": variable_encoder,
                                "token_expander": token_expander,
                                "params": params,
                                "stage": stage,
                                "input_mesh":input_mesh}

    # ------------------------------------- INIT INFO ---------------------------------------
    hyperparameters = {
        "Foundation FNO": [0.85, 0.9, 0.95, 0.97,0.98,0.99],
        "Foundation Codano": [0.75, 0.8, 0.85, 0.9, 0.95] # large
    }
    results_by_model = {'Foundation FNO': {}, 'Foundation Codano': {}, 'Foundation Codano Linear':{}}

# ================================= RUN COMPARISON =======================================
    # Read
    with open("compression/LowRank/results/foundation_result2.pkl", "rb") as f:
        results_by_model = pickle.load(f)
    print(results_by_model)

    # # ------------------------------------- FOUNDATION FNO ---------------------------------------
    # print("<"+"="*50, "Processing Foundation FNO", 50*"="+">")
    # foundation_fnos = []
    # ffno_hyperparams = hyperparameters["Foundation FNO"]
    # for ratio in ffno_hyperparams:
    #     ffnolowrank_model = CompressedModel(
    #         model=ffno_model,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio, is_compress_FC=False),
    #         create_replica=True
    #     )

    #     ffnolowrank_model = ffnolowrank_model.to(device)
    #     foundation_fnos.append(ffnolowrank_model)

    # ffnocompare = compare_models_hyperparams(
    #     model1=ffno_model,
    #     model2s=foundation_fnos,
    #     hyperparameters = ffno_hyperparams,
    #     test_loaders=validation_loaders_ffno,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True
    # )
    # results_by_model["Foundation FNO"]["Comparison"].update(ffnocompare["Comparison"])

    # # ------------------------------------- FOUNDATION CODANO ---------------------------------------
    # print("<"+"="*50, "Processing Foundation Codano", 50*"="+">")
    # foundation_codanos = []
    # fcodano_hyperparams = hyperparameters["Foundation Codano"]
    # for ratio in fcodano_hyperparams:
    #     fcodanolowrank_model = CompressedModel(
    #         model=fcodano_model,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio, is_compress_spectral=False, is_compress_conv1d=True),
    #         create_replica=True
    #     )
    #     fcodanolowrank_model = fcodanolowrank_model.to(device)
    #     foundation_codanos.append(fcodanolowrank_model)

    # fcodano_compare = compare_models_hyperparams(
    #     model1=fcodano_model,
    #     model2s=foundation_codanos,
    #     hyperparameters=fcodano_hyperparams,
    #     test_loaders=validation_loaders_fcodano,
    #     data_processor=None,
    #     device=device,
    #     track_performance = True,
    #     evaluation_params = fcodano_evaluation_params
    # )
    # results_by_model["Foundation Codano Linear"] = fcodano_compare

    # # ------------------------------------- Results Store ---------------------------------------
    # with open("compression/LowRank/results/foundation_result2.pkl", "wb") as f:
    #     pickle.dump(results_by_model, f)


    # # ------------------------------------- Final Evaluation ---------------------------------------
    # print(results_by_model)

    generate_graph(results_by_model, hyperparameters, "SVD_low_rank", "Rank Ratio", "%", savefile="compression/LowRank/results/foundation_lowrank_performance.png")