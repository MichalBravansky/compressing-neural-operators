import pickle
import random
import h5py
import os
import torch
import numpy as np
from torchvision.transforms import Normalize
from torch.utils.data import ConcatDataset, random_split, DataLoader, Dataset
import itertools
from neuralop.utils import *
from neuralop.data.datasets.codano_tensor_dataset import TensorDataset
from neuralop.data_utils.data_utils import get_mesh_displacement

import torch
import pickle
import argparse
import random
random.seed(42)

class TestDataset(Dataset):
    def __init__(self, test_data):
        """
        Args:
            test_data (list): List of test data batches.
        """
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        return self.test_data[idx]


def load_test_set(file_path):
    """
    Load the exported test set from a pickle file.
    
    Args:
        file_path: Path to the pickle file containing the test set
        
    Returns:
        test_data: List of test data batches
        metadata: Metadata about the test set
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    test_data = data['test_data']
    metadata = data['metadata']
    
    # print("Test set loaded successfully")
    # print(f"Configuration: {metadata['config']}")
    # print(f"Total samples: {metadata['total_samples']}")
    # print(f"Equations: {metadata['equation_dict']}")
    # print(f"Mu list: {metadata['mu_list']}")
    # shape of dataset: [{}, {}, {}]
    print(type(test_data))
    test_dataset = TestDataset(test_data)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    return test_data, test_dataloader, metadata

# dataloader for Fluid Sturctur Interaction (FSI) problems

class IrregularMeshTensorDataset(TensorDataset):
    def __init__(
            self,
            x,
            y,
            transform_x=None,
            transform_y=None,
            equation=None,
            x1=0,
            x2=0,
            mu=0.1,
            mesh=None):
        super().__init__(x, y, transform_x, transform_y)
        self.x1 = x1
        self.x2 = x2

        self.mu = mu
        self.mesh = mesh
        self.equation = equation
        print("Inside Dataset :", self.mesh.dtype, x.dtype, x.dtype)
        self._creat_static_features()

    def _creat_static_features(self, d_grid=None):
        '''
        creating static channels for inlet and reynolds number
        '''
        n_grid_points = self.x.shape[1]
        if len(self.equation) == 1:
            # equation can be either  ['NS'] or ['NS', 'ES']
            # of 3 or 5 channels/varibales
            n_variables = 3
        else:
            n_variables = self.x.shape[-1]
        if d_grid is not None:
            positional_enco = self.mesh + d_grid
        else:
            positional_enco = self.mesh

        raynolds = torch.ones(n_grid_points, 1) * self.mu
        inlet = ((-self.x1 / 2 + positional_enco[:, 1]) *
                 (-self.x2 / 2 + positional_enco[:, 1]))[:, None]**2

        self.static_features = torch.cat(
            [raynolds, inlet, positional_enco], dim=-1).repeat(1, n_variables)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        d_grid_x = get_mesh_displacement(x)
        d_grid_y = get_mesh_displacement(y)

        self._creat_static_features(d_grid_x)

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        if len(self.equation) == 1:
            x = x[:, :3]
            y = y[:, :3]

        return {'x': x, 'y': y, 'd_grid_x': d_grid_x,
                'd_grid_y': d_grid_y, 'static_features': self.static_features,
                'equation': self.equation}


class Normalizer():
    def __init__(self, mean, std, eps=1e-6, persample=False):
        self.persample = persample
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        if self.persample:
            self.mean = torch.mean(data, dim=(0))
            self.std = torch.var(data, dim=(0))**0.5
        return (data - self.mean) / (self.std + self.eps)

    def denormalize(self, data):
        return data * (self.std + self.eps) + self.mean

    def cuda(self,):
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()


class NsElasticDataset():
    def __init__(self, location, equation, mesh_location, params):
        self.location = location

        # _x1 and _x2 are the paraemters for the inlets condtions
        # _mu is the visocity
        self._x1 = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        self._x2 = [-4.0, -2.0, 0, 2.0, 4.0, 6.0]
        self._mu = [0.1, 0.01, 0.5, 5, 1, 10]
        if params.data_partition == 'supervised':
            # held out 2 inlets for finetuning
            # there not introduced in the self-supevised
            # pretraining
            self._x1 = params.supervised_inlets_x1
            self._x2 = params.supervised_inlets_x2
        elif params.data_partition == 'self-supervised':
            self._x1 = list(set(self._x1) - set(params.supervised_inlets_x1))
            self._x2 = list(set(self._x2) - set(params.supervised_inlets_x2))
        else:
            raise ValueError(
                f"Data partition {params.data_partition} not supported")

        self.equation = equation

        mesh = get_mesh(params)
        self.input_mesh = torch.from_numpy(mesh).type(torch.float)
        print("Mesh Shape: ", self.input_mesh.shape)
        self.params = params

        self.normalizer = Normalizer(None, None, persample=True)

    def _readh5(self, h5f, dtype=torch.float32):
        a_dset_keys = list(h5f['VisualisationVector'].keys())
        size = len(a_dset_keys)
        readings = [None for i in range(size)]
        for dset in a_dset_keys:
            ds_data = (h5f['VisualisationVector'][dset])
            readings[int(dset)] = torch.tensor(np.array(ds_data), dtype=dtype)

        readings_tensor = torch.stack(readings, dim=0)
        print(f"Loaded tensor Size: {readings_tensor.shape}")
        return readings_tensor

    def get_data(self, mu, x1, x2):
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if x1 not in self._x1 or x2 not in self._x2:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} ")
        if mu == 0.5:
            path = os.path.join(
                self.location,
                'mu=' + str(mu),
                'x1=' + str(-4.0),
                'x2=' + str(x2),
                '1',
                'Visualization')
            print(path)
        else:
            path = os.path.join(
                self.location,
                'mu=' + str(mu),
                'x1=' + str(x1),
                'x2=' + str(x2),
                'Visualization')

        filename = os.path.join(path, 'displacement.h5')

        h5f = h5py.File(filename, 'r')
        displacements_tensor = self._readh5(h5f)

        filename = os.path.join(path, 'pressure.h5')
        h5f = h5py.File(filename, 'r')
        pressure_tensor = self._readh5(h5f)

        filename = os.path.join(path, 'velocity.h5')
        h5f = h5py.File(filename, 'r')
        velocity_tensor = self._readh5(h5f)

        return velocity_tensor, pressure_tensor, displacements_tensor

    def get_data_txt(self, mu, x1, x2):
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if x1 not in self._x1 or x2 not in self._x2:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} ")
        path = os.path.join(
            self.location,
            'mu=' + str(mu),
            'x1=' + str(x1),
            'x2=' + str(x2),
            '1')

        velocity_x = torch.tensor(np.loadtxt(os.path.join(path, 'vel_x.txt')))
        velocity_y = torch.tensor(np.loadtxt(os.path.join(path, 'vel_y.txt')))
        if len(self.params.equation_dict) != 1:
            dis_x = torch.tensor(np.loadtxt(os.path.join(path, 'dis_x.txt')))
            dis_y = torch.tensor(np.loadtxt(os.path.join(path, 'dis_y.txt')))
            pressure = torch.tensor(np.loadtxt(os.path.join(path, 'pres.txt')))
        else:
            # just copying values as place holder when only NS equation is used
            dis_x = velocity_x
            dis_y = velocity_y
            pressure = velocity_x

        # reshape each tensor into 2d by keeping 876 entries in each row
        dis_x = dis_x.view(-1, 876, 1)
        dis_y = dis_y.view(-1, 876, 1)
        pressure = pressure.view(-1, 876, 1)
        velocity_x = velocity_x.view(-1, 876, 1)
        velocity_y = velocity_y.view(-1, 876, 1)

        velocity = torch.cat([velocity_x, velocity_y], dim=-1)
        displacement = torch.cat([dis_x, dis_y], dim=-1)

        return velocity.to(
            torch.float), pressure.to(
            torch.float), displacement.to(
            torch.float)

    def get_validation_test_dataloader(
            self,
            mu_list,
            dt,
            normalize=True,
            batch_size=1,
            train_test_split=0.2,
            val_ratio=0.2, 
            sample_per_inlet=200,
            ntrain=None,
            ntest=None,
            data_loader_kwargs={'num_workers': 2}):

        train_datasets = []
        test_datasets = []

        for mu in mu_list:
            train, test = self.get_tensor_dataset(
                mu, dt, normalize, train_test_split=train_test_split, sample_per_inlet=sample_per_inlet)
            train_datasets.append(train)
            test_datasets.append(test)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

        if ntrain is not None:
            train_dataset = random_split(train_dataset, [ntrain, len(train_dataset) - ntrain])[0]

        total_test_size = len(test_dataset)
        test_dataset, _ = random_split(test_dataset, [500, total_test_size-500])

        total_test_size = len(test_dataset)
        val_size = int(total_test_size * val_ratio)
        test_size = total_test_size - val_size

        print("****Train dataset size***: ", len(train_dataset))
        print("***Test dataset size***: ", len(test_dataset))

        val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

        if ntest is not None:
            test_dataset = random_split(test_dataset, [ntest, len(test_dataset) - ntest])[0]

        train_dataloader = {128: DataLoader(train_dataset, batch_size=batch_size, **data_loader_kwargs)}
        val_dataloader = {128: DataLoader(val_dataset, batch_size=batch_size, **data_loader_kwargs)}
        test_dataloader = {128: DataLoader(test_dataset, batch_size=batch_size, **data_loader_kwargs)}

        return val_dataloader, test_dataloader

    def get_dataloader(
            self,
            mu_list,
            dt,
            normalize=True,
            batch_size=1,
            train_test_split=0.2,
            sample_per_inlet=200,
            ntrain=None,
            ntest=None,
            data_loader_kwargs={'num_workers': 2}):

        train_datasets = []
        test_datasets = []

        for mu in mu_list:
            train, test = self.get_tensor_dataset(
                mu, dt, normalize, train_test_split=train_test_split, sample_per_inlet=sample_per_inlet)
            train_datasets.append(train)
            test_datasets.append(test)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
        print("****Train dataset size***: ", len(train_dataset))
        print("***Test dataset size***: ", len(test_dataset))
        if ntrain is not None:
            train_dataset = random_split(
                train_dataset, [ntrain, len(train_dataset) - ntrain])[0]
        if ntest is not None:
            test_dataset = random_split(
                test_dataset, [ntest, len(test_dataset) - ntest])[0]

        train_dataloader = {128: DataLoader(
            train_dataset, batch_size=batch_size, **data_loader_kwargs)}
        test_dataloader = {128: DataLoader(
            test_dataset, batch_size=batch_size, **data_loader_kwargs)}

        return train_dataloader, test_dataloader

    def get_tensor_dataset(
            self,
            mu,
            dt,
            normalize=True,
            min_max_normalize=False,
            train_test_split=0.2,
            sample_per_inlet=200,
            x1_list=None,
            x2_list=None):

        if x1_list is None:
            x1_list = self._x1
        if x2_list is None:
            x2_list = self._x2
        train_datasets = []
        test_datasets = []
        # for the given mu
        # loop over all given inlets
        for x1 in x1_list:
            for x2 in x2_list:
                try:
                    if mu == 0.5:
                        velocities, pressure, displacements = self.get_data_txt(
                            mu, x1, x2)
                    else:
                        velocities, pressure, displacements = self.get_data(
                            mu, x1, x2)
                except FileNotFoundError as e:
                    print(e)
                    print(
                        f"Original file not found for mu={mu}, x1={x1}, x2={x2}")
                    continue

                # keeping vx,xy, P, dx,dy
                varable_idices = [0, 1, 3, 4, 5]
                if mu == 0.5:
                    combined = torch.cat(
                        [velocities, pressure, displacements], dim=-1)[:sample_per_inlet, :, :]
                else:
                    combined = torch.cat(
                        [velocities, pressure, displacements], dim=-1)[:sample_per_inlet, :, varable_idices]

                if hasattr(
                        self.params,
                        'sub_sample_size') and self.params.sub_sample_size is not None:
                    mesh_size = combined.shape[1]
                    indexs = [i for i in range(mesh_size)]
                    np.random.seed(self.params.random_seed)
                    sub_indexs = np.random.choice(
                        indexs, self.params.sub_sample_size, replace=False)
                    combined = combined[:, sub_indexs, :]

                if self.params.super_resolution:
                    new_quieries = self.get_data_txt(
                        mu, x1, x2).to(dtype=combined.dtype)
                    new_quieries = new_quieries[:sample_per_inlet, :]

                    print("shape of old data", combined.shape)
                    print("shape of new data", new_quieries.shape)

                    combined = torch.cat([combined, new_quieries], dim=-2)
                    print("shape of combined data", combined.shape)

                step_t0 = combined[:-dt, ...]
                step_t1 = combined[dt:, ...]

                indexs = [i for i in range(step_t0.shape[0])]

                ntrain = int((1 - train_test_split) * len(indexs))
                ntest = len(indexs) - ntrain

                random.shuffle(indexs)
                train_t0, test_t0 = step_t0[indexs[:ntrain]
                                            ], step_t0[indexs[ntrain:ntrain + ntest]]
                train_t1, test_t1 = step_t1[indexs[:ntrain]
                                            ], step_t1[indexs[ntrain:ntrain + ntest]]

                if not normalize:
                    normalizer = None
                else:
                    if not min_max_normalize:
                        mean, var = torch.mean(train_t0, dim=(
                            0, 1)), torch.var(train_t0, dim=(0, 1))**0.5
                    else:
                        mean = torch.min(
                            train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]
                        var = torch.max(train_t0.view(-1,
                                                      train_t0.shape[-1]),
                                        dim=0)[0] - torch.min(train_t0.view(-1,
                                                                            train_t0.shape[-1]),
                                                              dim=0)[0]

                    normalizer = Normalizer(mean, var)

                train_datasets.append(
                    IrregularMeshTensorDataset(
                        train_t0,
                        train_t1,
                        normalizer,
                        normalizer,
                        x1=x1,
                        x2=x2,
                        mu=mu,
                        equation=self.equation,
                        mesh=self.input_mesh))
                test_datasets.append(
                    IrregularMeshTensorDataset(
                        test_t0,
                        test_t1,
                        normalizer,
                        normalizer,
                        x1=x1,
                        x2=x2,
                        mu=mu,
                        equation=self.equation,
                        mesh=self.input_mesh))

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)