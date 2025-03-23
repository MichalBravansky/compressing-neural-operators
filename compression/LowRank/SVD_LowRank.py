from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Union, Tuple, Optional
from neuralop.models import FNO
from neuralop.losses import LpLoss, H1Loss
from neuralop.data.datasets import load_darcy_flow_small
import numpy as np
from neuralop.data.transforms.data_processors import MGPatchingDataProcessor
from neuralop.training.training_state import load_training_state
from neuralop.layers.spectral_convolution_lowrank import DoubleSpectralConv
from neuralop.layers.foundation_fno_layers_lowrank import SpectralConv2dV2_lowrank
from neuralop.layers.fino_2D_lowrank import SpectralConvKernel2d_lowrank
from neuralop.layers.fino_2D import SpectralConvKernel2d
import torch
import torch.nn as nn
from typing import Dict
from compression.base import CompressionTechnique
import copy
from tltorch.factorized_tensors.core import FactorizedTensor

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List


class SVDLowRank:
    """
    Handles low-rank decomposition for:
    - Conv1d(kernel_size=1) → Replaced with two smaller Conv1d layers.
    - SpectralConv (DenseTensor) → Performs low-rank SVD on channel dimensions.
    """
    def __init__(self, 
                 model, 
                 rank_ratio=0.5, 
                 min_rank=1, 
                 max_rank=1028,
                 is_full_rank = False,
                 is_compress_conv1d=False, 
                 is_compress_spectral=True,
                 is_compress_FC=True):
        
        self.rank_ratio = rank_ratio  
        self.min_rank = min_rank     
        self.max_rank = max_rank      
        self.model = model
        self.original_params = 0
        self.compressed_params = 0
        self.is_full_rank = is_full_rank
        self.is_compress_conv1d = is_compress_conv1d
        self.is_compress_spectral = is_compress_spectral
        self.is_compress_FC = is_compress_FC
        self.compressed_layers = {}
    
    def _get_target_rank(self, weight: torch.Tensor) -> int:
        if torch.isnan(weight).any() or not torch.isfinite(weight).all():
            raise ValueError("NaN/Inf")

        if weight.dim() != 2:
            weight = weight.reshape(weight.size(0), -1)
            #weight = weight.view(weight.size(0), -1) 
        
        weight = weight.cpu().float()

        _, S, _ = torch.linalg.svd(weight.float())
        energy = (S ** 2).cumsum(dim=0) / (S ** 2).sum()
        valid_indices = torch.where(energy <= self.rank_ratio)[0]
        rank = valid_indices.numel() + 1 if valid_indices.numel() > 0 else 1
        return max(min(rank, self.max_rank), self.min_rank)

    def _get_unified_rank(self, W, sample_num=4):
        """
        Sample a few frequency slices from W (shape: [modes, modes, C_in, C_out])
        and compute their target ranks using _get_target_rank.
        Return the maximum rank from the samples.
        """
        modes1, modes2, _, _ = W.shape
        sample_indices = [(i, j) for i in range(0, modes1, max(1, modes1//sample_num))
                            for j in range(0, modes2, max(1, modes2//sample_num))]
        ranks = []
        for i, j in sample_indices:
            slice_ij = W[i, j]  # shape: (C_in, C_out)
            r_temp = self._get_target_rank(slice_ij)
            ranks.append(r_temp)
        return max(ranks) if ranks else 64

    
    def compress_FC(self, layer, name):
        original_weight = layer.weight.data
        weight = torch.nan_to_num(original_weight, nan=0.0, posinf=1.0, neginf=-1.0)

        device = original_weight.device
        weight = weight.float().to(device)

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        if self.is_full_rank:
            rank = min(layer.in_features, layer.out_features)    
        else:
            rank = self._get_target_rank(weight=weight)
        # truncate from rank
        U_trunc = U[:, :rank]
        S_trunc = S[:rank]
        V_trunc = Vh[:rank, :]
        
        # create 2 weights
        weight1 = V_trunc
        weight2 = U_trunc @ torch.diag(S_trunc)  # [out_features, target_rank]

        weight1 = weight1.to(device)
        weight2 = weight2.to(device)

        total_n_params = weight1.numel() + weight2.numel()

        if (total_n_params >= original_weight.numel()):
            self.compressed_layers[name] = layer
        else:
            # create 2 linear layers
            linear1 = nn.Linear(layer.in_features, rank, bias=False)
            linear2 = nn.Linear(rank, layer.out_features, bias=layer.bias is not None)
            with torch.no_grad():
                linear1.weight.copy_(weight1)
                linear2.weight.copy_(weight2)
        
                # copy bias if there is
                if layer.bias is not None:
                    linear2.bias.copy_(layer.bias)
        
            # create a sequential
            seq = nn.Sequential(linear1, linear2)
            self.compressed_layers[name] = seq

    def compress_conv1d(self, layer, name):
        """
        Compress a Conv1d(kernel_size=1) layer using SVD decomposition.
        Replaces the layer with: Conv1d(in_channels, rank, 1) → Conv1d(rank, out_channels, 1) → ReLU
        """
        W = layer.weight.data  # [out_channels, in_channels, 1]
        out_channels, in_channels, kernel_size = W.shape
        
        if kernel_size != 1:
            raise ValueError(f"Layer {name} is Conv1d but kernel_size != 1. Skipping.")

        # Reshape to (out_channels, in_channels)
        W_2d = W.view(out_channels, in_channels)

        # Perform SVD
        U, S, Vh = torch.linalg.svd(W_2d, full_matrices=False)

        # Truncate to rank
        if self.is_full_rank:
            rank = min(layer.in_features, layer.out_features)    
        else:
            rank = self._get_target_rank(W_2d)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        weight1 = Vh.unsqueeze(2)     # [rank, in_channels, 1]
        weight2 = (U*S).unsqueeze(2)  # [out_channels, rank, 1]
        total_n_params = weight1.numel() + weight2.numel()

        if (total_n_params >= W.numel()):
            self.compressed_layers[name] = layer
        else:
            # Create two smaller Conv1d layers
            conv1 = nn.Conv1d(in_channels, rank, kernel_size=1, bias=False)
            conv2 = nn.Conv1d(rank, out_channels, kernel_size=1, bias=layer.bias is not None)
            # Assign new weights
            with torch.no_grad():
                conv1.weight.copy_(weight1)
                conv2.weight.copy_(weight2)
                # copy bias if there is
                if layer.bias is not None:
                    conv2.bias.copy_(layer.bias)

            class CompressedSequential(nn.Sequential):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args)
                    self.out_channels = kwargs.get("out_channels")
            seq = CompressedSequential(conv1, conv2, out_channels=out_channels)
            self.compressed_layers[name] = seq

    def compress_spectral_conv(self, layer, name):
        """
        Compress a SpectralConv layer by applying low-rank SVD on channel dimensions.
        """
        if not hasattr(layer.weight, "to_tensor"):
            print(f"[Warning] DenseTensor at {name} has no 'to_tensor()' method. Skipping.")
            return
        # Extract tensor
        original_weight = layer.weight.to_tensor()
        C_in, C_out, H, W = original_weight.shape
        weight = original_weight.permute(1,2,3,0).reshape(C_out*H*W, C_in)
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        if self.is_full_rank:
            _, rank = U.shape 
        else:
            rank = self._get_target_rank(weight)
        U_trunc = U[:, :rank]
        S_trunc = S[:rank].to(torch.complex64)
        Vh_trunc = Vh[:rank, :]

        weight1 = (Vh_trunc.T).reshape(C_in, rank, 1, 1).expand(-1, -1, H, W)
        weight2 = (U_trunc @ torch.diag(S_trunc)).reshape(C_out, H, W, rank).permute(3, 0, 1, 2)
        total_n_params = weight1.numel() + weight2.numel()
        if (total_n_params >= original_weight.numel()):
            self.compressed_layers[name] = layer
        else:
            new_layer = DoubleSpectralConv(in_channels=C_in,
                                        out_channels=C_out,
                                        n_modes=(H,W),
                                        mid_channels=rank)
            with torch.no_grad():
                new_layer.weight1 = nn.Parameter(weight1.clone(), requires_grad=True)
                new_layer.weight2 = nn.Parameter(weight2.clone(), requires_grad=True)
                new_layer.bias = layer.bias
            self.compressed_layers[name] = new_layer


    def compress_foundation_spectral_conv(self, layer, name):
        """
        Compress the foundation spectral convolution layer using low-rank approximation.
        The original weight has shape: (C_in, C_out, modes, modes, 2).
        This implementation avoids explicit m, n loops by merging the frequency dimensions
        and performing batched SVD. Note that a unified truncation rank is required for all frequency slices.
        """
        fac_weight1 = True
        fac_weight2 = True

        # Process weight1 (for branch 1)
        weight1 = layer.weights1  # shape: (C_in, C_out, modes1, modes2, 2)
        C_in, C_out, modes1, modes2, _ = weight1.shape

        # Convert weight1 to a complex tensor and permute dimensions to merge frequency dimensions:
        weight1_complex = torch.view_as_complex(weight1)  # shape: (C_in, C_out, modes1, modes2)
        W1 = weight1_complex.permute(2, 3, 0, 1)            # shape: (modes1, modes2, C_in, C_out)
        
        # Set a unified truncation rank for branch1 (must be the same for all frequency slices)
        # TODO:: auto-adapt depending on rank_ratio
        # final_rank1 = 64  # or: final_rank1 = self._get_target_rank(W1.reshape(modes1*modes2, C_in, C_out))
        final_rank1 = self._get_unified_rank(W1)
        
        # Batched SVD for weight1:
        U1, S1, Vh1 = torch.linalg.svd(W1, full_matrices=False)
        # U1: (modes1, modes2, C_in, min(C_in,C_out))
        # S1: (modes1, modes2, min(C_in,C_out))
        # Vh1: (modes1, modes2, min(C_in,C_out), C_out)
        
        # Truncate to final_rank1:
        U1_trunc = U1[:, :, :, :final_rank1]         # (modes1, modes2, C_in, final_rank1)
        S1_trunc = S1[:, :, :final_rank1]              # (modes1, modes2, final_rank1)
        Vh1_trunc = Vh1[:, :, :final_rank1, :]           # (modes1, modes2, final_rank1, C_out)
        
        # Scale U1 by singular values (broadcasting S1_trunc)
        U1_trunc_scaled = U1_trunc * S1_trunc.unsqueeze(-2)  # (modes1, modes2, C_in, final_rank1)
        
        # Rearrange dimensions to obtain U1 and V1 factors with expected shapes:
        # Expected U1: (C_in, final_rank1, modes1, modes2, 2)
        U1_factor = U1_trunc_scaled.permute(2, 3, 0, 1)  # (C_in, final_rank1, modes1, modes2)
        U1_factor = torch.view_as_real(U1_factor)         # (C_in, final_rank1, modes1, modes2, 2)
        
        # Expected V1: (final_rank1, C_out, modes1, modes2, 2)
        V1_factor = Vh1_trunc.permute(2, 3, 0, 1)         # (final_rank1, C_out, modes1, modes2)
        V1_factor = torch.view_as_real(V1_factor.resolve_conj())  # Resolve conjugation before conversion

        total_n_params1 = U1_factor.numel() + V1_factor.numel()

        # Process weight2 (for branch 2)
        weight2 = layer.weights2  # shape: (C2_in, C2_out, modes1_b, modes2_b, 2)
        C2_in, C2_out, modes1_b, modes2_b, _ = weight2.shape

        weight2_complex = torch.view_as_complex(weight2)   # shape: (C2_in, C2_out, modes1_b, modes2_b)
        W2 = weight2_complex.permute(2, 3, 0, 1)             # shape: (modes1_b, modes2_b, C2_in, C2_out)
        
        # Set a unified truncation rank for branch2:
        final_rank2 = self._get_unified_rank(W2)
        
        # Batched SVD for weight2:
        U2, S2, Vh2 = torch.linalg.svd(W2, full_matrices=False)
        # U2: (modes1_b, modes2_b, C2_in, min(C2_in,C2_out))
        # S2: (modes1_b, modes2_b, min(C2_in,C2_out))
        # Vh2: (modes1_b, modes2_b, min(C2_in,C2_out), C2_out)
        
        U2_trunc = U2[:, :, :, :final_rank2]         # (modes1_b, modes2_b, C2_in, final_rank2)
        S2_trunc = S2[:, :, :final_rank2]              # (modes1_b, modes2_b, final_rank2)
        Vh2_trunc = Vh2[:, :, :final_rank2, :]           # (modes1_b, modes2_b, final_rank2, C2_out)
        
        U2_trunc_scaled = U2_trunc * S2_trunc.unsqueeze(-2)  # (modes1_b, modes2_b, C2_in, final_rank2)
        
        # Rearrange dimensions to obtain U2 and V2 factors:
        # Expected U2: (C2_in, final_rank2, modes1_b, modes2_b, 2)
        U2_factor = U2_trunc_scaled.permute(2, 3, 0, 1)  # (C2_in, final_rank2, modes1_b, modes2_b)
        U2_factor = torch.view_as_real(U2_factor)         # (C2_in, final_rank2, modes1_b, modes2_b, 2)
        
        # Expected V2: (final_rank2, C2_out, modes1_b, modes2_b, 2)
        V2_factor = Vh2_trunc.permute(2, 3, 0, 1)         # (final_rank2, C2_out, modes1_b, modes2_b)
        V2_factor = torch.view_as_real(V2_factor.resolve_conj())
        total_n_params2 = U2_factor.numel() + V2_factor.numel()

        # replace the layer
        if (total_n_params1 >= weight1.numel()):
            print("yes")
            fac_weight1 = False
        if (total_n_params2 >= weight2.numel()):
            print("yes")
            fac_weight2 = False
        # Create the new low-rank spectral convolution layer.
        # Here, the new layer expects unified ranks for both branches.
        new_layer = SpectralConv2dV2_lowrank(in_channels=C_in,
                                            out_channels=C_out,
                                            modes1=modes1,
                                            modes2=modes2,
                                            rank1=final_rank1,
                                            rank2=final_rank2,
                                            fac_weight1=fac_weight1,
                                            fac_weight2=fac_weight2)
        if fac_weight1:    
            new_layer.U1 = nn.Parameter(U1_factor)
            new_layer.V1 = nn.Parameter(V1_factor)
        else:
            new_layer.weights1 = layer.weights1
        if fac_weight2:
            new_layer.U2 = nn.Parameter(U2_factor)
            new_layer.V2 = nn.Parameter(V2_factor)
        else:
            new_layer.weights2 = layer.weights2
            
        # If the original layer has a bias, assign it accordingly (if needed)
        # new_layer.bias = layer.bias

        # Save the compressed layer
        self.compressed_layers[name] = new_layer

    # for foundation codano
    # large tensor tried
    def compress_spectral_conv_2d_kernel(self, layer, name):
        """
        Compress a SpectralConv layer by applying low-rank SVD on channel dimensions.
        """
        weights = [] 
        new_ranks = []     
        for weight in layer.weight:
            original_weight = weight.to_tensor()
            C_in, C_out, H, W = original_weight.shape
            weight = original_weight.permute(1,2,3,0).reshape(C_out*H*W, C_in)
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
            if self.is_full_rank:
                _, rank = U.shape 
            else:
                rank = self._get_target_rank(weight)

            U_trunc = U[:, :rank]
            S_trunc = S[:rank].to(torch.complex64)
            Vh_trunc = Vh[:rank, :]


            weight1 = (Vh_trunc.T).reshape(C_in, rank, 1, 1).expand(-1, -1, H, W)

            weight2 = (U_trunc @ torch.diag(S_trunc)).reshape(C_out, H, W, rank).permute(3, 0, 1, 2)

            total_n_params = weight1.numel() + weight2.numel()
            if total_n_params > original_weight.numel():
                weights.append(original_weight)
                new_ranks.append(0)
            else:
                weights.append((weight1, weight2))
                new_ranks.append(rank)


        new_layer = SpectralConvKernel2d_lowrank(in_channels=C_in,
                                    out_channels=C_out,
                                    n_modes=layer.n_modes,
                                    ranks=new_ranks,
                                    n_layers=layer.n_layers,
                                    output_scaling_factor=layer.output_scaling_factor,
                                    rank=layer.rank,
                                    factorization=layer.factorization,
                                    implementation=layer.implementation,
                                    transform_type=layer.transform_type)
        
        with torch.no_grad():
            for i in range(len(new_ranks)):
                if new_ranks[i] != 0:
                    weight1, weight2 = weights[i]
                    new_layer.weight[i][0].tensor.data.copy_(weight1)
                    new_layer.weight[i][1].tensor.data.copy_(weight2)
                else:
                    new_layer.weight[i].tensor.data.copy_(weights[i])
            new_layer.bias = layer.bias
        self.compressed_layers[name] = new_layer


    def compress(self):
        """
        Iterates through the model and applies low-rank decomposition.
        - Conv1d(kernel_size=1) is replaced with two smaller Conv1d layers.
        - SpectralConv (DenseTensor) is approximated using low-rank SVD.
        """
        self.original_params = sum(p.numel() for p in self.model.parameters())
        for name, module in self.model.named_modules():
            if self.is_compress_conv1d and isinstance(module, nn.Conv1d): #and module.kernel_size == (1,):
                self.compress_conv1d(module, name)
            elif self.is_compress_FC and isinstance(module, nn.Linear):
                self.compress_FC(module, name)
            # Handle SpectralConv (DenseTensor) 
            elif self.is_compress_spectral and ("SpectralConv" == type(module).__name__):
                if hasattr(module, "weight"):
                   # print(1)
                    #print(module.weight.shape)
                    self.compress_spectral_conv(module, name)
            elif self.is_compress_spectral and ("SpectralConv2dV2" == type(module).__name__):
                self.compress_foundation_spectral_conv(module,name)
            elif self.is_compress_spectral and ("SpectralConvKernel2d" == type(module).__name__):
                self.compress_spectral_conv_2d_kernel(module, name)

        # Replace original layers with compressed versions
        for name, new_layer in self.compressed_layers.items():
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent_module = self.model
            if parent_name:
                for part in parent_name.split("."):
                    parent_module = getattr(parent_module, part)
            setattr(parent_module, child_name, new_layer)

        self.compressed_params = sum(p.numel() for p in self.model.parameters())
        print("[LowRank] Compression applied successfully.")
        print("Original:",self.original_params)
        print("Compressed", self.compressed_params)
        return self.model

    def get_compression_stats(self) -> Dict[str, Union[int, float]]:
        # Compute compression ratio & sparsity
        compression_ratio = self.compressed_params / self.original_params
        sparsity = 1 - compression_ratio        
        return {
            "original_parameters": self.original_params,
            "compressed_parameters": self.compressed_params,
            "compression_ratio": compression_ratio,
            "sparsity": sparsity,
            "compressed_layers": list(self.compressed_layers.keys())
        }
