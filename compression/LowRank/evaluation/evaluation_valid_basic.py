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

    # ------------------------------------- INIT CODANO MODEL ---------------------------------------
    codano_model = CODANO(
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


    codano_model.load_model(torch.load("models/model-codano-darcy-16-resolution-2025-03-15-19-31.pt", weights_only=False))
    codano_model.eval()
    codano_model = codano_model.to(device)

    validation_loaders_codano, test_loaders_codano, data_processor_codano = load_darcy_flow_small_validation_test(
        n_train=10000,
        batch_size=16,
        test_resolutions=[128],
        n_tests=[1000],
        test_batch_sizes=[4, 4],
        encode_input=False, 
        encode_output=False,
    )

    # When creating data processor, use the imported class
    data_processor_codano = CODANODataProcessor(
        in_normalizer=data_processor_codano.in_normalizer,
        out_normalizer=data_processor_codano.out_normalizer
    )


    # # ------------------------------------- INIT DEEPONET MODEL ---------------------------------------
    # deeponet_model = DeepONet(
    # train_resolution=128,
    # in_channels=1,
    # out_channels=1, 
    # hidden_channels=64,
    # branch_layers=[256, 256, 256, 256, 128],
    # trunk_layers=[256, 256, 256, 256, 128],
    # positional_embedding='grid',
    # non_linearity='gelu',
    # norm='instance_norm',
    # dropout=0.1
    # )

    # deeponet_model.load_state_dict(torch.load("models/model-deeponet-darcy-128-resolution-2025-03-04-18-53.pt", weights_only=False))
    # deeponet_model.eval()
    # deeponet_model = deeponet_model.to(device)

    # validation_loaders_deeponet, test_loaders_deeponet, data_processor_deeponet = load_darcy_flow_small_validation_test(
    #     n_train=10000,
    #     batch_size=16,
    #     test_resolutions=[128],
    #     n_tests=[1000],
    #     test_batch_sizes=[4, 4],
    #     encode_input=False, 
    #     encode_output=False,
    # )

    # # ------------------------------------- INIT FNO 16x16 MODEL ---------------------------------------
    # fno_model_16, validation_loaders_fno16, test_loaders_fno16, data_processor_fno16 = optional_fno(resolution="low")
    # fno_model_16 = fno_model_16.to(device)

    # # ------------------------------------- INIT FNO 32x32 MODEL ---------------------------------------
    # fno_model_32, validation_loaders_fno32, test_loaders_fno32, data_processor_fno32 = optional_fno(resolution="medium")
    # fno_model_16 = fno_model_16.to(device)

    # # ------------------------------------- INIT FNO 128x128 MODEL ---------------------------------------
    # fno_model_128, validation_loaders_fno128, test_loaders_fno128, data_processor_fno128 = optional_fno(resolution="high")
    # fno_model_128 = fno_model_128.to(device)



    # ------------------------------------- INIT INFO ---------------------------------------
    hyperparameters = {
        "FNO 16x16": [0.55, 0.56, 0.57, 0.58, 0.6], # small
        "FNO 32x32": [0.4,0.45,0.47, 0.5, 0.55],
        "Codano":  [0.5, 0.55, 0.6, 0.65, 0.7],
        "FNO 128x128": [0.45, 0.5, 0.51, 0.52, 0.53],
        "DeepONet": [0.95, 0.96, 0.97, 0.98, 0.99],
    }
    results_by_model = {'FNO 16x16': {}, 'FNO 32x32': {}, 'FNO 128x128': {}, 'Codano': {}, 'DeepONet': {}}

# ================================= RUN COMPARISON =======================================
    # Read
    with open("compression/LowRank/results/basic_result2.pkl", "rb") as f:
        results_by_model = pickle.load(f)
    print(results_by_model)

# ------------------------------------- CODANO ---------------------------------------
    print("<"+"="*50, "Processing CODANO", 50*"="+">")
    codanos = []
    codano_hyperparams = hyperparameters["Codano"]
    for ratio in codano_hyperparams:
        codanolowrank_model = CompressedModel(
            model=codano_model,
            compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio,                                                           
                                                           is_compress_conv1d=True,
                                                           is_compress_spectral=False),
            create_replica=True
        )
        codanolowrank_model = codanolowrank_model.to(device)
        codanos.append(codanolowrank_model)

    codano_compare = compare_models_hyperparams(
        model1=codano_model,
        model2s=codanos,
        hyperparameters=codano_hyperparams,
        test_loaders=validation_loaders_codano,
        data_processor=data_processor_codano,
        device=device,
        track_performance=True
    )
    results_by_model["Codano"] = codano_compare

    # # ------------------------------------- DeepONet ---------------------------------------
    # print("<"+"="*50, "Processing DeepONet", 50*"="+">")
    # deeponets = []
    # deeponet_hyperparams = hyperparameters["DeepONet"]
    # for ratio in deeponet_hyperparams:
    #     deepolowrank_model = CompressedModel(
    #         model=deeponet_model,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
    #         create_replica=True
    #     )
    #     deepolowrank_model = deepolowrank_model.to(device)
    #     deeponets.append(deepolowrank_model)

    # deepo_compare = compare_models_hyperparams(
    #     model1=deeponet_model,
    #     model2s=deeponets,
    #     hyperparameters=deeponet_hyperparams,
    #     test_loaders=validation_loaders_deeponet,
    #     data_processor=data_processor_deeponet,
    #     device=device,
    #     track_performance = True
    # )
    # results_by_model["DeepONet"] = deepo_compare

    # # ------------------------------------- FNO 16 ---------------------------------------
    # print("<"+"="*50, "Processing FNO 16x16", 50*"="+">")
    # fno_16s = []
    # fno16_hyperparams = hyperparameters["FNO 16x16"]
    # for ratio in fno16_hyperparams:
    #     fnolowrank_model_16 = CompressedModel(
    #         model=fno_model_16,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
    #         create_replica=True
    #     )
    #     fnolowrank_model_16 = fnolowrank_model_16.to(device)
    #     fno_16s.append(fnolowrank_model_16)

    # fnocompare_16 = compare_models_hyperparams(
    #     model1=fno_model_16,
    #     model2s=fno_16s,
    #     hyperparameters=fno16_hyperparams,
    #     test_loaders=validation_loaders_fno16,
    #     data_processor=data_processor_fno16,
    #     device=device,
    #     track_performance=True
    # )
    # results_by_model["FNO 16x16"].update(fnocompare_16)

    # # ------------------------------------- FNO 32 ---------------------------------------
    # print("<"+"="*50, "Processing FNO 32x32", 50*"="+">")
    # fno_32s = []
    # fno32_hyperparams = hyperparameters["FNO 32x32"]
    # for ratio in fno32_hyperparams:
    #     fnolowrank_model_32 = CompressedModel(
    #         model=fno_model_32,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
    #         create_replica=True
    #     )
    #     fnolowrank_model_32 = fnolowrank_model_32.to(device)
    #     fno_32s.append(fnolowrank_model_32)

    # fnocompare_32 = compare_models_hyperparams(
    #     model1=fno_model_32,
    #     model2s=fno_32s,
    #     hyperparameters=fno32_hyperparams,
    #     test_loaders=validation_loaders_fno32,
    #     data_processor=data_processor_fno32,
    #     device=device,
    #     track_performance=True
    # )
    # results_by_model["FNO 32x32"] = fnocompare_32

    # # ------------------------------------- FNO 128 ---------------------------------------
    # print("<"+"="*50, "Processing FNO 128x128", 50*"="+">")
    # fno_128s = []
    # fno128_hyperparams = hyperparameters["FNO 128x128"]
    # for ratio in fno128_hyperparams:
    #     fnolowrank_model_128 = CompressedModel(
    #         model=fno_model_128,
    #         compression_technique=lambda model: SVDLowRank(model, rank_ratio=ratio),
    #         create_replica=True
    #     )
    #     fnolowrank_model_128 = fnolowrank_model_128.to(device)
    #     fno_128s.append(fnolowrank_model_128)

    # fnocompare_128 = compare_models_hyperparams(
    #     model1=fno_model_128,
    #     model2s=fno_128s,
    #     hyperparameters=fno128_hyperparams,
    #     test_loaders=validation_loaders_fno128,
    #     data_processor=data_processor_fno128,
    #     device=device,
    #     track_performance=True
    # )
    # results_by_model["FNO 128x128"] = fnocompare_128

    # ------------------------------------- Results Store ---------------------------------------
    """
`   "FNO 16x16": [0.45, 0.5, 0.55, 0.58, 0.6], # small
    "FNO 32x32": [0.4,0.45,0.47, 0.5, 0.55],
    "Codano":  [0.5, 0.55, 0.6, 0.65, 0.7],
    "FNO 128x128": [0.45, 0.5, 0.51, 0.52, 0.53],
    "DeepONet": [0.95, 0.96, 0.97, 0.98, 0.99]
    -> basic_result1.pkl

    "FNO 16x16": [0.55, 0.56, 0.57, 0.58, 0.6], # small
    "FNO 32x32": [0.4,0.45,0.47, 0.5, 0.55],
    "Codano":  [0.5, 0.55, 0.6, 0.65, 0.7],
    "FNO 128x128": [0.45, 0.5, 0.51, 0.52, 0.53],
    "DeepONet": [0.95, 0.96, 0.97, 0.98, 0.99],
    -> basic_result2.pkl
    """

    #Write
    with open("compression/LowRank/results/basic_result2.pkl", "wb") as f:
        pickle.dump(results_by_model, f)


    # ------------------------------------- Final Evaluation ---------------------------------------
    generate_graph(results_by_model, hyperparameters, "SVD_low_rank", "Rank Ratio", "%", savefile="compression/LowRank/results/basic_lowrank_performance.png")