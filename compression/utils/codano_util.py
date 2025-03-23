from neuralop.data_utils.data_utils import *
from neuralop.models.get_models import *
from ruamel.yaml import YAML
from timeit import default_timer
import time
from tqdm import tqdm


class ParamsBase:
    """Convenience wrapper around a dictionary

    Allows referring to dictionary items as attributes, and tracking which
    attributes are modified.
    """

    def __init__(self):
        self._original_attrs = None
        self.params = {}
        self._original_attrs = list(self.__dict__)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def to_dict(self):
        new_attrs = {
            key: val for key, val in vars(self).items()
            if key not in self._original_attrs
        }
        return {**self.params, **new_attrs}

    @staticmethod
    def from_json(path: str) -> "ParamsBase":
        with open(path) as f:
            c = json.load(f)
        params = ParamsBase()
        params.update_params(c)
        return params

    def update_params(self, config):
        for key, val in config.items():
            if val == 'None':
                val = None

            if type(val) == dict:
                child = ParamsBase()
                child.update_params(val)
                val = child

            self.params[key] = val
            self.__setattr__(key, val)


class CodanoYParams(ParamsBase):
    def __init__(self, yaml_filename, config_name, print_params=False):
        """Open parameters stored with ``config_name`` in the yaml file ``yaml_filename``"""
        super().__init__()
        self._yaml_filename = yaml_filename
        self._config_name = config_name

        with open(yaml_filename) as _file:
            d = YAML().load(_file)[config_name]

        self.update_params(d)

        if print_params:
            print("------------------ Configuration ------------------")
            for k, v in d.items():
                print(k, end='=')
                print(v)
            print("---------------------------------------------------")

    def log(self):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.to_dict().items():
            logging.info(str(key) + ' ' + str(val))
        logging.info("---------------------------------------------------")

# from codano trainer
def get_grid_displacement(params, stage, data):
    if params.grid_type == "non uniform":
        with torch.no_grad():
            if stage == StageEnum.RECONSTRUCTIVE:
                out_grid_displacement = data['d_grid_x'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
            else:
                out_grid_displacement = data['d_grid_y'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
    else:
        out_grid_displacement = None
        in_grid_displacement = None
    return out_grid_displacement, in_grid_displacement

# codano evaluation

def missing_variable_testing(
        model,
        test_loader,
        augmenter=None,
        stage=StageEnum.PREDICTIVE,
        params=None,
        variable_encoder=None,
        token_expander=None,
        initial_mesh=None,
        wandb_log=False):
    print('Evaluating for Stage: ', stage)
    model.eval()
    with torch.no_grad():
        ntest = 0
        test_l2 = 0
        test_l1 = 0
        loss_p = nn.MSELoss()
        loss_l1 = nn.L1Loss()
        t1 = default_timer()
        predictions = []
        for data in tqdm(test_loader):
            x, y = data['x'].cuda(), data['y'].cuda()
            static_features = data['static_features']

            if augmenter is not None:
                x, _ = batched_masker(x, augmenter)

            inp = prepare_input(
                x,
                static_features,
                params,
                variable_encoder,
                token_expander,
                initial_mesh,
                data)
            out_grid_displacement, in_grid_displacement = get_grid_displacement(
                params, stage, data)

            batch_size = x.shape[0]
            out = model(inp, out_grid_displacement=out_grid_displacement,
                        in_grid_displacement=in_grid_displacement)

            if getattr(params, 'horizontal_skip', False):
                out = out + x

            if isinstance(out, (list, tuple)):
                out = out[0]

            ntest += 1
            target = y.clone()

            predictions.append((out, target))

            test_l2 += loss_p(target.reshape(batch_size, -1),
                              out.reshape(batch_size, -1)).item()
            test_l1 += loss_l1(target.reshape(batch_size, -1),
                               out.reshape(batch_size, -1)).item()

    test_l2 /= ntest
    test_l1 /= ntest
    t2 = default_timer()
    avg_time = (t2 - t1) / ntest

    print(f"Augmented Test Error  {stage}: ", test_l2)

    # if hasattr(params, 'save_predictions') and params.save_predictions:
    #     torch.save(predictions[:50], f'../xy/predictions_{params.config}.pt')

def get_grid_displacement(params, stage, data):
    if params.grid_type == "non uniform":
        with torch.no_grad():
            if stage == StageEnum.RECONSTRUCTIVE:
                out_grid_displacement = data['d_grid_x'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
            else:
                out_grid_displacement = data['d_grid_y'].cuda()[0]
                in_grid_displacement = data['d_grid_x'].cuda()[0]
    else:
        out_grid_displacement = None
        in_grid_displacement = None
    return out_grid_displacement, in_grid_displacement