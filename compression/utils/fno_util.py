from ruamel.yaml import YAML
import logging
from neuralop.models.fno import FNO
import torch
from compression.magnitude_pruning.global_pruning import GlobalMagnitudePruning
from compression.LowRank.SVD_LowRank import SVDLowRank
from compression.quantization.dynamic_quantization import DynamicQuantization
from compression.base import CompressedModel
from neuralop.data.datasets.darcy import load_darcy_flow_small_validation_test

class FNOYParams():
  """ Yaml file parser """
  def __init__(self, yaml_filename, config_name, print_params=False):
    self._yaml_filename = yaml_filename
    self._config_name = config_name
    self.params = {}

    if print_params:
      print("------------------ Configuration ------------------")

    with open(yaml_filename, encoding='utf-8') as _file:

      for key, val in YAML().load(_file)[config_name].items():
        if print_params: print(key, val)
        if val =='None': val = None

        self.params[key] = val
        self.__setattr__(key, val)

    if print_params:
      print("---------------------------------------------------")

  def __getitem__(self, key):
    return self.params[key]

  def __setitem__(self, key, val):
    self.params[key] = val
    self.__setattr__(key, val)

  def __contains__(self, key):
    return (key in self.params)

  def update_params(self, config):
    for key, val in config.items():
      self.params[key] = val
      self.__setattr__(key, val)

  def log(self):
    logging.info("------------------ Configuration ------------------")
    logging.info("Configuration file: "+str(self._yaml_filename))
    logging.info("Configuration name: "+str(self._config_name))
    for key, val in self.params.items():
        logging.info(str(key) + ' ' + str(val))
    logging.info("---------------------------------------------------")

def optional_fno(resolution):
    # Low resolution FNO
    if resolution == "low":
        fno_model = FNO(
            in_channels=1,
            out_channels=1,
            n_modes=(16, 16),
            hidden_channels=16,
            projection_channel_ratio=2,
            n_layers=5,
            skip="linear",
            norm="group_norm",
            implementation="factorized",
            separable=False,
            factorization=None,
            rank=1.0,
            domain_padding=None,
            stabilizer=None,
            dropout=0.0)
        
        fno_model.load_state_dict(torch.load("models/model-fno-darcy-16-resolution-2025-03-17-18-57.pt", weights_only=False))
        fno_model.eval()    
        validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
            n_train=1000,
            batch_size=16,
            test_resolutions=[16],
            n_tests=[10000],
            test_batch_sizes=[16],
            encode_input=False, 
            encode_output=False,
        )
        return fno_model, validation_loaders, test_loaders, data_processor
    
    elif resolution == "medium":
        fno_model = FNO(
        in_channels=1,
        out_channels=1,
        n_modes=(32, 32),
        hidden_channels=32,
        projection_channel_ratio=2,
        n_layers=5,
        skip="linear",
        norm="group_norm",
        implementation="factorized",
        separable=False,
        factorization=None,
        rank=1.0,
        domain_padding=None,
        stabilizer=None,
        dropout=0.0)

        fno_model.load_state_dict(torch.load("models/model-fno-darcy-16-resolution-2025-03-17-19-02.pt", weights_only=False))
        fno_model.eval()

        validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
            n_train=100,
            batch_size=16,
            test_resolutions=[32],
            n_tests=[10000],
            test_batch_sizes=[16],
            encode_input=True, 
            encode_output=False,
        )
        return fno_model, validation_loaders, test_loaders, data_processor
    
    elif resolution == "high":
        fno_model = FNO(
        in_channels=1,
        out_channels=1,
        n_modes=(32, 32),
        hidden_channels=64,
        projection_channel_ratio=2,
        n_layers=5,
        skip="linear",
        norm="group_norm",
        implementation="factorized",
        separable=False,
        factorization=None,
        rank=1.0,
        domain_padding=None,
        stabilizer=None,
        dropout=0.0)

        fno_model.load_state_dict(torch.load("models/model-fno-darcy-16-resolution-2025-03-04-18-48.pt", weights_only=False))
        fno_model.eval()

        validation_loaders, test_loaders, data_processor = load_darcy_flow_small_validation_test(
            n_train=10000,
            batch_size=16,
            test_resolutions=[128],
            n_tests=[1000],
            test_batch_sizes=[16],
            encode_input=True, 
            encode_output=False,
        )
        return fno_model, validation_loaders, test_loaders, data_processor