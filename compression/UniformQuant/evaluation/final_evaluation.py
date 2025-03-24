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
from compression.UniformQuant.uniform_quant import UniformQuantisation

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

# TODO: DONT forget to change back the val_ratio of FOUNDATION CODANO
if __name__ == "__main__":

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # ------------------------------------- INIT FOUNDATION FNO MODEL ---------------------------------------
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--yaml_config", default='config/mixed_config.yaml', type=str)
    # parser.add_argument("--config", default='fno-foundational', type=str)
    # parser.add_argument("--root_dir", default='./', type=str, help='root dir to store results')
    # parser.add_argument("--run_num", default='0', type=str, help='sub run config')
    # parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    # args = parser.parse_args()
    # params = FNOYParams(os.path.abspath(args.yaml_config), args.config, print_params=False)
    # if dist.is_initialized():
    #     dist.barrier()
    # logging.info('DONE')
    # params['global_batch_size'] = params.batch_size
    # params['local_batch_size'] = int(params.batch_size)

    # params['global_valid_batch_size'] = params.valid_batch_size
    # params['local_valid_batch_size'] = int(params.valid_batch_size)
    # ffno_model = fno(params).to(device)
    # checkpoints = torch.load("models/ckpt_best.tar", map_location='cpu', weights_only=False)  
    # ffno_model.load_state_dict(checkpoints['model_state'])
    # ffno_model.eval()

    # validation_loaders_ffno, test_loaders_ffno, data_processor_ffno = get_data_val_test_loader(params,
    #                                                                 params.test_path, 
    #                                                                 dist.is_initialized(), 
    #                                                                 train=False, 
    #                                                                 pack=params.pack_data)

    # # ------------------------------------- INIT FOUNDATION CODANO MODEL ---------------------------------------
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp", nargs="?", default="FSI", type=str)
    # parser.add_argument("--config", nargs="?", default="codano_gno_NS_ES", type=str)
    # args = parser.parse_args()
    # config_file = './config/ssl_ns_elastic.yaml'
    # print("Loading config", args.config)
    # params = CodanoYParams(config_file, args.config, print_params=True)
    # torch.manual_seed(params.random_seed)
    # random.seed(params.random_seed)
    # np.random.seed(params.random_seed)
    # params.config = args.config
    # stage = StageEnum.PREDICTIVE
    # variable_encoder = None
    # token_expander = None

    # encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    # if params.use_variable_encoding:
    #     variable_encoder = get_variable_encoder(params)
    #     token_expander = TokenExpansion(
    #         sum([params.equation_dict[i] for i in params.equation_dict.keys()]),
    #         params.n_encoding_channels, params.n_static_channels,
    #         params.grid_type == 'uniform'
    #     )

    # fcodano_model = SSLWrapper(params, encoder, decoder, contrastive, predictor, stage=stage)

    # print("Setting the Grid")
    # mesh = get_mesh(params)
    # input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()
    # fcodano_model.set_initial_mesh(input_mesh)

    # stage = StageEnum.PREDICTIVE
    # fcodano_model.load_state_dict(torch.load(params.model_path), strict=False)
    # fcodano_model = fcodano_model.cuda().eval()

    # variable_encoder.load_encoder(
    #                 "NS", params.NS_variable_encoder_path)

    # variable_encoder.load_encoder(
    #             "ES", params.ES_variable_encoder_path)

    # if variable_encoder is not None:
    #     variable_encoder = variable_encoder.cuda().eval()
    # if token_expander is not None:
    #     token_expander = token_expander.cuda().eval()

    # dataset_fcodano = NsElasticDataset(
    #     params.data_location,
    #     equation=list(params.equation_dict.keys()),
    #     mesh_location=params.input_mesh_location,
    #     params=params
    # )

    # validation_loaders_fcodano, test_loaders_fcodano = dataset_fcodano.get_validation_test_dataloader(
    #     params.mu_list, params.dt,
    #     ntrain=params.get('ntrain'),
    #     ntest=40,
    #     batch_size = 1,
    #     #val_ratio=0.01,
    #     sample_per_inlet=params.sample_per_inlet
    # )

    # grid_non, grid_uni = get_meshes(params, params.grid_size)
    # test_augmenter = None

    # fcodano_evaluation_params = {"variable_encoder": variable_encoder,
    #                             "token_expander": token_expander,
    #                             "params": params,
    #                             "stage": stage,
    #                             "input_mesh":input_mesh}

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
        n_train=10,
        batch_size=16,
        test_resolutions=[16],
        n_tests=[10, 5],
        test_batch_sizes=[16, 16],
        encode_input=False, 
        encode_output=False,
    )

    # When creating data processor, use the imported class
    data_processor_codano = CODANODataProcessor(
        in_normalizer=data_processor_codano.in_normalizer,
        out_normalizer=data_processor_codano.out_normalizer
    )


    # ------------------------------------- INIT DEEPONET MODEL ---------------------------------------
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

    # deeponet_model.load_state_dict(torch.load("models/model-deeponet-darcy-128-resolution-2025-02-19-22-23.pt", weights_only=False))
    # deeponet_model.eval()
    # deeponet_model = deeponet_model.to(device)

    # validation_loaders_deeponet, test_loaders_deeponet, data_processor_deeponet = load_darcy_flow_small_validation_test(
    #     n_train=10,
    #     batch_size=4,
    #     test_resolutions=[128],
    #     n_tests=[10],
    #     test_batch_sizes=[4, 4],
    #     encode_input=False, 
    #     encode_output=False,
    # )

    # ------------------------------------- INIT FNO 16x16 MODEL ---------------------------------------
    # fno_model_16, validation_loaders_fno16, test_loaders_fno16, data_processor_fno16 = optional_fno(resolution="low")
    # fno_model_16 = fno_model_16.to(device)

    # ------------------------------------- INIT FNO 32x32 MODEL ---------------------------------------
    # fno_model_32, validation_loaders_fno32, test_loaders_fno32, data_processor_fno32 = optional_fno(resolution="medium")
    # fno_model_32 = fno_model_32.to(device)

    # # ------------------------------------- INIT FNO 128x128 MODEL ---------------------------------------
    # fno_model_128, validation_loaders_fno128, test_loaders_fno128, data_processor_fno128 = optional_fno(resolution="high")
    # fno_model_128 = fno_model_128.to(device)



    # ------------------------------------- INIT INFO ---------------------------------------
    hyperparameters = {
        "FNO 16x16": [8], # small
        "FNO 32x32": [8],
        "Codano":  [8],
        "FNO 128x128": [8], # medium
        "DeepONet": [8],
        "Foundation FNO": [8],
        "Foundation Codano": [8] # large
    }
    number_of_hyperparameters = 1
    results_by_model = {'FNO 16x16': {}, 'FNO 32x32': {}, 'FNO 128x128': {}, 'Codano': {},
                        'DeepONet': {}, 'Foundation FNO': {}, 'Foundation Codano': {}}

    ## ------------------------------------- RUN COMPARISON ---------------------------------------

    # # ------------------------------------- FOUNDATION FNO ---------------------------------------
    #     foundation_fnos = []
    #     ffno_hyperparams = hyperparameters["Foundation FNO"]
    #     for num_bits in ffno_hyperparams:
    #         ffnoquant_model = CompressedModel(
    #             model=ffno_model,
    #             compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits),
    #             create_replica=True
    #         )

    #         ffnoquant_model = ffnoquant_model.to(device)
    #         foundation_fnos.append(ffnoquant_model)

    #     ffnocompare = compare_models_hyperparams(
    #         model1=ffno_model,
    #         model2s=ffnoquant_model,
    #         hyperparameters = ffno_hyperparams,
    #         test_loaders=validation_loaders_ffno,
    #         data_processor=None,
    #         device=device,
    #         track_performance = True
    #     )
    #     results_by_model["Foundation FNO"] = ffnocompare

    # # ------------------------------------- FOUNDATION CODANO ---------------------------------------
    #     foundation_codanos = []
    #     fcodano_hyperparams = hyperparameters["Foundation Codano"]
    #     for num_bits in fcodano_hyperparams:
    #         fcodanoquant_model = CompressedModel(
    #             model=fcodano_model,
    #             compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits),
    #             create_replica=True
    #         )
    #         fcodanoquant_model = fcodanoquant_model.to(device)
    #         foundation_codanos.append(fcodanoquant_model)

    #     fcodano_compare = compare_models_hyperparams(
    #         model1=fcodano_model,
    #         model2s=foundation_codanos,
    #         hyperparameters=fcodano_hyperparams,
    #         test_loaders=validation_loaders_fcodano,
    #         data_processor=None,
    #         device=device,
    #         track_performance = True,
    #         evaluation_params = fcodano_evaluation_params
    #     )
    #     results_by_model["Foundation Codano"] = fcodano_compare

    # # ------------------------------------- CODANO ---------------------------------------
    codanos = []
    codano_hyperparams = hyperparameters["Codano"]
    for num_bits in codano_hyperparams:
        codanoquant_model = CompressedModel(
            model=codano_model,
            compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits),
            create_replica=True
        )
        codanoquant_model = codanoquant_model.to(device)
        codanos.append(codanoquant_model)

    codano_compare = compare_models_hyperparams(
        model1=codano_model,
        model2s=codanos,
        hyperparameters=codano_hyperparams,
        test_loaders=validation_loaders_codano,
        data_processor=data_processor_codano,
        device=device,
        #track_performance=True
    )
    results_by_model["Codano"] = codano_compare

    # # ------------------------------------- DeepONet ---------------------------------------
    # deeponets = []
    # deeponet_hyperparams = hyperparameters["DeepONet"]
    # for num_bits in deeponet_hyperparams:
    #     deepoquant_model = CompressedModel(
    #         model=deeponet_model,
    #         compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits, num_calibration_runs=512),
    #         create_replica=True
    #     )
    #     deepoquant_model = deepoquant_model.to(device)
    #     deeponets.append(deepoquant_model)

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

    # ------------------------------------- FNO 16 ---------------------------------------
    # fno_16s = []
    # fno16_hyperparams = hyperparameters["FNO 16x16"]
    # for num_bits in fno16_hyperparams:
    #     fnoquantised_model_16 = CompressedModel(
    #         model=fno_model_16,
    #         compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits, num_calibration_runs=1),
    #         create_replica=True
    #     )
    #     fnoquantised_model_16 = fnoquantised_model_16.to(device)
    #     fno_16s.append(fnoquantised_model_16)

    # fnocompare_16 = compare_models_hyperparams(
    #     model1=fno_model_16,
    #     model2s=fno_16s,
    #     hyperparameters=fno16_hyperparams,
    #     test_loaders=validation_loaders_fno16,
    #     data_processor=data_processor_fno16,
    #     device=device,
    #     #track_performance=True
    # )
    # results_by_model["FNO 16x16"].update(fnocompare_16)

    # # ------------------------------------- FNO 32 ---------------------------------------
    # fno_32s = []
    # fno32_hyperparams = hyperparameters["FNO 32x32"]
    # for num_bits in fno32_hyperparams:
    #     fnoquant_model_32 = CompressedModel(
    #         model=fno_model_32,
    #         compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits, num_calibration_runs=512),
    #         create_replica=True
    #     )
    #     fnoquant_model_32 = fnoquant_model_32.to(device)
    #     fno_32s.append(fnoquant_model_32)

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
    # fno_128s = []
    # fno128_hyperparams = hyperparameters["FNO 128x128"]
    # for num_bits in fno128_hyperparams:
    #     fnoquant_model_128 = CompressedModel(
    #         model=fno_model_128,
    #         compression_technique=lambda model: UniformQuantisation(model, num_bits=num_bits),
    #         create_replica=True
    #     )
    #     fnoquant_model_128 = fnoquant_model_128.to(device)
    #     fno_128s.append(fnoquant_model_128)

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
    print("dumping results")
    with open("compression/UniformQuant/results/result.pkl", "wb") as f:
        pickle.dump(results_by_model, f)

    # # Read
    # with open("compression/staticquant/results/result.pkl", "rb") as f:
    #     results_by_model = pickle.load(f)
    # print(results_by_model)
    # ------------------------------------- Final Evaluation ---------------------------------------\
    print(results_by_model)

    generate_graph(results_by_model, hyperparameters, "Static_quantisation", "Number Bits Used", "b", savefile="compression/UniformQuant/results/all_staticquant_performance.png")