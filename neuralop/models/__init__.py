from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
from .local_no import LocalNO
# only import SFNO if torch_harmonics is built locally
try:
    from .sfno import SFNO
except ModuleNotFoundError:
    pass
from .uno import UNO
from .uqno import UQNO
from .fnogno import FNOGNO
from .gino import GINO
from .codano import CODANO
from .deeponet import DeepONet
from .base_model import get_model
