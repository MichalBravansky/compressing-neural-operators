from typing import List, Optional, Union
from math import prod
from pathlib import Path
import torch

# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )

def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2*n_params
    return n_params


def wandb_login(api_key_file="config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.
    normalize: bool
        whether to apply normalization to the output of the 2D FFT. 
        If True, normalizes the outputs by ``1/n_observations``
        (actually ``1/sqrt(n_observations * n_observations)``). 
    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
        The spectrum is computed using a square approximation to radial
        binning, meaning that the wavenumber 'bin' into which a particular 
        coefficient is the coefficient's location along the diagonal, indexed 
        from the top-left corner of the 2d FFT output. 
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal, norm="ortho")
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum


Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None

def compute_rank(tensor):
    # Compute the matrix rank of a tensor
    rank = torch.matrix_rank(tensor)
    return rank

def compute_stable_rank(tensor):
    # Compute the stable rank of a tensor
    tensor = tensor.detach()
    fro_norm = torch.linalg.norm(tensor, ord='fro')**2
    l2_norm = torch.linalg.norm(tensor, ord=2)**2
    rank = fro_norm / l2_norm
    rank = rank
    return rank

def compute_explained_variance(frequency_max, s):
    # Compute the explained variance based on frequency_max and singular
    # values (s)
    s_current = s.clone()
    s_current[frequency_max:] = 0
    return 1 - torch.var(s - s_current) / torch.var(s)

def get_project_root():
    root = Path(__file__).parent.parent
    return root


# from codano
import datetime
import logging
import os
import pathlib
import psutil
import re
import signal

from typing import List

import h5py
# from haikunator import Haikunator
import numpy as np
import psutil
import torch
import torch.nn as nn

# HAIKU = haikunator.Haikunator()


def prepare_input(
        x,
        static_features,
        params,
        variable_encoder,
        token_expander,
        initial_mesh,
        data):
    if variable_encoder is not None and token_expander is not None:
        if params.grid_type == 'uniform':
            inp = token_expander(x, variable_encoder(x),
                                 static_features.cuda())
        elif params.grid_type == 'non uniform':
            initial_mesh = initial_mesh.cuda()
            #quation = [i[0] for i in data['equation']]
            equation = [i[0] if isinstance(i[0], str) else i[0][0] for i in data['equation']]
            inp = token_expander(
                x, 
                variable_encoder(
                    initial_mesh +
                    data['d_grid_x'].cuda()[0],
                    equation),
                static_features.cuda())
    elif params.n_static_channels > 0 and params.grid_type == 'non uniform':
        inp = torch.cat(
            [x, static_features[:, :, :params.n_static_channels].cuda()], dim=-1)
    else:
        inp = x
    return inp


def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()

def get_mesh(params):
    """Get the mesh from a location."""
    if hasattr(params, "text_mesh") and params.text_mesh:
        # load mesh_x and mesh_y from txt file as np array
        mesh_x = np.loadtxt(params.mesh_x)
        mesh_y = np.loadtxt(params.mesh_y)
        # create mesh from mesh_x and mesh_y
        mesh = np.zeros((mesh_x.shape[0], 2))
        mesh[:, 0] = mesh_x
        mesh[:, 1] = mesh_y
    else:
        h5f = h5py.File(params.input_mesh_location, 'r')
        mesh = h5f['mesh/coordinates']

    if params.super_resolution:
        # load mesh_x and mesh_y from txt file as np array
        mesh_x = np.loadtxt(params.super_resolution_mesh_x)
        mesh_y = np.loadtxt(params.super_resolution_mesh_y)
        # create mesh from mesh_x and mesh_y
        mesh_sup = np.zeros((mesh_x.shape[0], 2))
        mesh_sup[:, 0] = mesh_x
        mesh_sup[:, 1] = mesh_y
        # merge it with the original mesh
        mesh = np.concatenate((mesh, mesh_sup), axis=0)

        print("Super Resolution Mesh Shape", mesh.shape)

    if hasattr(
            params,
            'sub_sample_size') and params.sub_sample_size is not None:
        mesh_size = mesh.shape[0]
        indexs = [i for i in range(mesh_size)]
        np.random.seed(params.random_seed)
        sub_indexs = np.random.choice(
            indexs, params.sub_sample_size, replace=False)
        mesh = mesh[sub_indexs, :]

    return mesh[:]


# TODO add collision checks
# TODO add opts to toggle haiku and date fixes
def save_model(
        model,
        directory: pathlib.Path,
        stage=None,
        sep='_',
        name=None,
        config=None):
    """Saves a model with a unique prefix/suffix

    The model is prefixed with is date (formatted like YYMMDDHHmm)
    and suffixed with a "Heroku-like" name (for verbal reference).

    Params:
    ---
    stage: None | StageEnum
        Controls the infix of the model name according to the following mapping:
        | None -> "model"
        | RECONSTRUCTIVE -> "reconstructive"
        | PREDICTIVE -> "predictive"
    """
    prefix = datetime.datetime.utcnow().strftime("%y%m%d%H%M")
    infix = "model"
    if stage is not None:
        infix = stage.value.lower()
    # suffix = Haikunator.haikunate(token_length=0, delimiter=sep)

    torch.save(model.state_dict(), directory / f"{name}{sep}{config}{sep}.pth")


def extract_pids(message) -> List[int]:
    # Assume `message` has a preamble followed by a sequence of tokens like
    # "Process \d+" with extra characters in between such tokens.

    pattern = re.compile("(Process \\d+)")
    # Contains "Process" tokens and extra characters, interleaved:
    tokens = pattern.split(message)
    # print('\n'.join(map(repr, zip(split[1::2], split[2::2]))))

    pattern2 = re.compile("(\\d+)")
    # print('\n'.join([repr((s, pattern2.search(t)[0])) for t in tokens[1::2]]))
    pids = [int(pattern2.search(t)[0]) for t in tokens[1::2]]

    return pids


# https://psutil.readthedocs.io/en/latest/#kill-process-tree
def signal_process_tree(
    pid,
    sig=signal.SIGTERM,
    include_parent=True,
    timeout=None,
    on_terminate=None,
    logger=None,
):
    """Kill a process tree (including grandchildren) with signal ``sig``

    Return a (gone, still_alive) tuple.

    Parameters
    ---
    timeout: float
        Time, in seconds, to wait on each signaled process.
    on_terminate: Optional[Callable]
        A callback function which is called as soon as a child terminates.
        Optional.
    """
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    if logger is None:
        logger = logging.getLogger()

    wait_children = []
    for p in children:
        try:
            p.send_signal(sig)
            wait_children.append(p)
        except psutil.AccessDenied:
            logger.error(f"Unable to terminate Process {pid} (AccessDenied)")
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(
        wait_children,
        timeout=timeout,
        callback=on_terminate,
    )
    return (gone, alive)


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model

    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel()
         for p in model.parameters()]
    )


def signal_my_processes(
    username,
    pids,
    sig=signal.SIGTERM,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger()
    my_pids = []
    for pid in pids:
        p = psutil.Process(pid)
        with p.oneshot():
            p = p.as_dict(attrs=["username", "status"])

        # TODO add other states to the filter
        if p["username"] == username and p["status"] == "sleeping":
            my_pids.append(pid)
        else:
            _p = {"pid": pid, **p}
            logger.warning(f"Cannot signal process: {_p}")

    for my_pid in my_pids:
        gone, alive = signal_process_tree(pid, sig, timeout=60, logger=logger)
        logger.info(f"{gone=}, {alive=}")