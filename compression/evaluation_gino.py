from neuralop import get_model
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.UniformQuant.uniform_quant import UniformQuantisation
from compression.base import CompressedModel
from neuralop.data.datasets import CarCFDDataset
from compression.utils import evaluate_model, compare_models
from configmypy import ConfigPipeline, YamlConfig
from neuralop.data.transforms.data_processors import DataProcessor
from copy import deepcopy
from neuralop.data.transforms.gino_processor import GINOCFDDataProcessor
from compression.quantization.dynamic_quantization import DynamicQuantization

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

config_name = 'cfd'
pipe = ConfigPipeline([YamlConfig('./gino_carcfd_config.yaml', config_name=config_name, config_folder='./config')])
config = pipe.read_conf()

gino_model = get_model(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gino_model.load_state_dict(torch.load("models/model-gino-carcfd-32-resolution-2025-02-12-18-47.pt", weights_only=False))
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

pruned_model = CompressedModel(
    model=gino_model,
    compression_technique=lambda model: GlobalMagnitudePruning(model, prune_ratio=0.01),
    create_replica=True
)
pruned_model = pruned_model.to(device)

lowrank_model = CompressedModel(
    model=gino_model,
    compression_technique=lambda model: SVDLowRank(model, rank_ratio=0.7, 
                                                   min_rank=8, max_rank=16),
    create_replica=True
)
lowrank_model = lowrank_model.to(device)

# Compare models
compare_models(
    model1=gino_model,
    model2=pruned_model,
    test_loaders={'test': test_loader},
    data_processor=data_processor,
    device=device
)