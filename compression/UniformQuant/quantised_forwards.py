import torch
from neuralop.models.fno import FNO
from neuralop.layers.fno_block import *
from neuralop.layers.skip_connections import *
from neuralop.layers.coda_layer import CODALayer
from neuralop.models.deeponet import DeepONet
from neuralop.models.codano import CODANO
from neuralop.layers.resample import resample
import numpy as np

def true_quantize(x : torch.Tensor):
    if not x.is_quantized:
        return torch.quantize_per_tensor_dynamic(x, torch.quint8, False)
    return x

def true_dequantize(x : torch.Tensor):
    if x.is_quantized:
        return x.dequantize()
    return x

def quantised_gating_forward(self : SoftGating, x):
        """Applies soft-gating to a batch of activations"""
        if self.bias is not None:
            return self.weight * true_dequantize(x) + self.bias
        else:
            return self.weight * true_dequantize(x)

def quantised_forward_with_postactivation(self : FNOBlocks, x: torch.Tensor, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](true_quantize(x))
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](x)
        x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](x, output_shape=output_shape)
        #self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](true_quantize(x_fno))

        x = true_quantize(true_dequantize(x_fno) + true_dequantize(x_skip_fno))

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = true_quantize(true_dequantize(self.channel_mlp[index](x)) + x_skip_channel_mlp)

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

def quantised_forward_with_preactivation(self, x, index=0, output_shape=None):
    # Apply non-linear activation (and norm)
    # before this block's convolution/forward pass:
    x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index](x)

    x_skip_fno = self.fno_skips[index](x)
    x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

    x_skip_channel_mlp = self.channel_mlp_skips[index](x)
    x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

    if self.stabilizer == "tanh":
        if self.complex_data:
            x = ctanh(x)
        else:
            x = torch.tanh(x)

    x_fno = self.convs[index](x, output_shape=output_shape)

    x = x_fno + x_skip_fno

    if index < (self.n_layers - 1):
        x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index + 1](x)

    x = self.channel_mlp[index](x) + x_skip_channel_mlp

    return x

def quantised_fno_forward(self : FNO, x: torch.Tensor, output_shape=None, **kwargs):
        """FNO's forward pass
        
        1. Applies optional positional encoding

        2. Sends inputs through a lifting layer to a high-dimensional latent space

        3. Applies optional domain padding to high-dimensional intermediate function representation

        4. Applies `n_layers` Fourier/FNO layers in sequence (SpectralConvolution + skip connections, nonlinearity) 

        5. If domain padding was applied, domain padding is removed

        6. Projection of intermediate function representation to the output channels

        Parameters
        ----------
        x : tensor
            input tensor
        
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            
            * If None, don't specify an output shape

            * If tuple, specifies the output-shape of the **last** FNO Block

            * If tuple list, specifies the exact output-shape of each FNO Block
        """
        model_device = next(self.parameters()).device
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]
        
        x = true_quantize(x)
        # append spatial pos embedding if set
        if self.positional_embedding is not None:
            x = true_dequantize(x)
            x = x.to(torch.device('cpu'))
            x = self.positional_embedding(x)
            x = x.to(model_device)
            x = true_quantize(x)
        
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = true_dequantize(x)
            x = self.domain_padding.pad(x)
            x = true_quantize(x)

        for layer_idx in range(self.n_layers):
            x = true_dequantize(x)
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
            x = true_quantize(x)

        if self.domain_padding is not None:
            x = true_dequantize(x)
            x = self.domain_padding.unpad(x)
            x = true_quantize(x)

        x = self.projection(x)

        return true_dequantize(x)

def quantised_deeponet_forward(self : DeepONet, x : torch.Tensor, y : torch.Tensor):
    """DeepONet forward pass
    
    Parameters
    ----------
    x : tensor
        Input function values, shape (batch, channels, height, width)
    y : tensor
        Coordinate points, shape (batch, channels, height, width)
        
    Returns
    -------
    tensor
        Output function values, shape (batch, out_channels, height, width)
    """
    x = true_quantize(x)
    y = true_quantize(y)
    batch_size, _, height, width = x.shape
    
    # Branch network on input function
    branch = x
    if self.branch_pos_embedding is not None:
        branch = true_quantize(self.branch_pos_embedding(true_dequantize(branch)))

    branch = branch.view(batch_size, -1)

    for i, layer in enumerate(self.branch_net):
        branch = layer(branch)
        if i < len(self.branch_net) - 1:
            branch = self.non_linearity(branch)
    
    # Trunk network on coordinates
    trunk = y
    if self.trunk_pos_embedding is not None:
        trunk = true_quantize(self.trunk_pos_embedding(true_dequantize(trunk)))

    trunk = trunk.view(batch_size, -1)
    
    for i, layer in enumerate(self.trunk_net):
        trunk = layer(trunk)
        if i < len(self.trunk_net) - 1:
            trunk = self.non_linearity(trunk)
    
    # Reshape branch and trunk to match spatial dimensions
    branch = branch.view(batch_size, -1, height, width)
    trunk = trunk.view(batch_size, -1, height, width)
    
    # Element-wise multiplication and sum across channel dimension
    out = true_quantize((true_dequantize(branch) * true_dequantize(trunk)).sum(dim=1, keepdim=True))
    
    # Add bias
    out = true_dequantize(out) + self.bias
    
    return out

def quantised_docano_forward(self : CODANO, **kwargs):
    """
    Parameters
    ----------
    x : torch.Tensor
        input tensor of shape (batch_size, num_inp_var, H, W, ...)
    static_channel : torch.Tensor
        static channel tensor of shape (batch_size, static_channel_dim, H, W, ...)
        These channels provide additional information regarding the physical setup of the system.
        Must be provided when `static_channel_dim > 0`.
    input_variable_ids : list[str]
        The names of the variables corresponding to the channels of input 'x'.
        This parameter is required when `use_positional_encoding=True`.

        For example, if input x represents and snapshot of the velocity field of a fluid flow, the variable_ids=['u_x', 'u_y'].
        The variable_ids must be in the same order as the channels in the input tensor 'x', i.e., variable_ids[0] corresponds to the
        first channel of 'x', i.e., x[:, 0, ...].

    Returns
    -------
    torch.Tensor
        output tensor of shape (batch_size, output_variable_codimension*num_inp_var, H, W, ...)
    """
    x = kwargs.get('in_data')
    static_channel = kwargs.get('static_channels')
    input_variable_ids = kwargs.get('variable_ids')

    batch, num_inp_var, *spatial_shape = (
        x.shape
    )  # num_inp_var is the number of channels in the input

    # input validation
    if (
        self.static_channel_dim > 0
        and static_channel is None
        and static_channel.shape[1] != self.static_channel_dim
    ):
        raise ValueError(
            f"Epected static channel dimension is {self.static_channel_dim}, but got {static_channel.shape[1]}"
        )
    if self.use_positional_encoding:
        assert (
            input_variable_ids is not None
        ), "variable_ids are not provided for the input"
        assert x.shape[1] == len(
            input_variable_ids
        ), f"Expected number of variables in input is {len(input_variable_ids)}, but got {x.shape[1]}"

    # position encoding and static channels are concatenated with the input
    # variables
    x = self._extend_variables(x, static_channel, input_variable_ids)

    # input variables are lifted to a higher-dimensional space
    if self.lifting:
        x = x.reshape(
            batch * num_inp_var, self.extended_variable_codimemsion, *spatial_shape
        )
        x = true_quantize(x)
        x = self.lifting(x)
        x = true_dequantize(x)
    x = x.reshape(
        batch, num_inp_var * self.hidden_variable_codimension, *spatial_shape
    )

    # getting the learnable CLASS token
    if self.enable_cls_token:
        cls_token = self._get_cls_token(x)
        x = torch.cat(
            [
                cls_token,
                x,
            ],
            dim=1,
        )
        num_inp_var += 1

    # zero padding the domain of the input
    if self.domain_padding is not None:
        x = self.domain_padding.pad(x)

    # calculating the output shape
    output_shape = [
        int(round(i * j))
        for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling)
    ]

    # forward pass through the Codomain Attention layers
    skip_outputs = {}
    for layer_idx in range(self.n_layers):

        if (
            self.horizontal_skips_map is not None
            and layer_idx in self.horizontal_skips_map.keys()
        ):
            # `horizontal skip connections`
            # tokens from skip connections are concatenated with the
            # current token and then linearly projected
            # to the `hidden_variable_codimension`
            skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
            resolution_scaling_factors = [
                m / n for (m, n) in zip(x.shape, skip_val.shape)
            ]
            resolution_scaling_factors = resolution_scaling_factors[
                -1 * self.n_dim :
            ]
            t = resample(
                skip_val,
                resolution_scaling_factors,
                list(range(-self.n_dim, 0)),
                output_shape=x.shape[-self.n_dim :],
            )
            x = x.reshape(
                batch * num_inp_var,
                self.hidden_variable_codimension,
                *x.shape[-self.n_dim :],
            )
            t = t.reshape(
                batch * num_inp_var,
                self.hidden_variable_codimension,
                *t.shape[-self.n_dim :],
            )
            x = torch.cat([x, t], dim=1)
            x = self.skip_map_module[str(layer_idx)](true_quantize(x))
            x = x.reshape(
                batch,
                num_inp_var * self.hidden_variable_codimension,
                *x.shape[-self.n_dim :],
            )

        if layer_idx == self.n_layers - 1:
            cur_output_shape = output_shape
        else:
            cur_output_shape = None


        x = self.attention_layers[layer_idx](x, output_shape=cur_output_shape)

        # storing the outputs for skip connections
        if (
            self.horizontal_skips_map is not None
            and layer_idx in self.horizontal_skips_map.values()
        ):
            skip_outputs[layer_idx] = x.clone()

    # removing the padding
    if self.domain_padding is not None:
        x = self.domain_padding.unpad(x)

    # projecting the hidden variables to the output variables
    if self.projection:
        x = x.reshape(
            batch * num_inp_var,
            self.hidden_variable_codimension,
            *x.shape[-self.n_dim :],
        )
        x = true_quantize(x)
        x = self.projection(x)
        x = true_dequantize(x)
        x = x.reshape(
            batch,
            num_inp_var * self.output_variable_codimension,
            *x.shape[-self.n_dim :],
        )
    else:
        return x

    # discarding the CLASS token
    if self.enable_cls_token:
        x = x[:, self.output_variable_codimension :, ...]
    return x

def quantised_docano_coda_layer_forward_equivariant(self : CODALayer, x : torch.Tensor, output_shape=None):
    """
    Forward pass with a permutation equivariant mixer layer after the
    attention mechanism. Shares the same mixer layer for all tokens, meaning
    that outputs are equivariant to permutations of the tokens.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (b, t * d, h, w, ...), where
        b is the batch size, t is the number of tokens, and d is the token codimension.
    """
    batch_size = true_dequantize(x).shape[0]
    input_shape = x.shape[-self.n_dim:]

    assert x.shape[1] % self.token_codimension == 0, "Number of channels in x should be divisible by token_codimension"

    # reshape from shape b (t*d) h w ... to (b*t) d h w ...
    t = x.size(1) // self.token_codimension
    tokens = x.reshape(
        x.size(0) * t,
        self.token_codimension,
        *x.shape[-self.n_dim:])

    # normalization and attention mechanism
    tokens = true_quantize(tokens)
    tokens_norm = self.norm1(tokens)
    attention = self.compute_attention(true_dequantize(tokens_norm), batch_size)
    if self.multi_head_proj is not None:
        attention = self.multi_head_proj(attention)
    attention = self.attention_normalizer(true_quantize(true_dequantize(attention) + true_dequantize(tokens)))

    # Pass through mixer layers sequentially
    output = self.mixer_in_normalizer(attention)
    for i in range(self.mixer.n_layers):
        output = self.mixer(true_dequantize(output), index=i, output_shape=input_shape)
    output = true_dequantize(self.mixer_out_normalizer(true_quantize(output))) + true_dequantize(attention)

    # reshape from shape (b*t) d h w... to b (t d) h w ...
    t = output.size(0) // batch_size
    output = output.view(
        batch_size,
        t * output.size(1),
        *output.shape[-self.n_dim:])
    
    if output_shape is not None:
        output = resample(output,
                            res_scale=[j/i for (i, j) in zip(output.shape[-self.n_dim:], output_shape)],
                            axis=list(range(-self.n_dim, 0)),
                            output_shape=output_shape)

    return output

def quantised_forward_with_postactivation_docano(self : FNOBlocks, x: torch.Tensor, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](true_quantize(x))
        x_skip_fno = self.convs[index].transform(true_dequantize(x_skip_fno), output_shape=output_shape)

        x_skip_channel_mlp = self.channel_mlp_skips[index](true_quantize(x))
        x_skip_channel_mlp = self.convs[index].transform(true_dequantize(x_skip_channel_mlp), output_shape=output_shape)

        if self.stabilizer == "tanh":
            if self.complex_data:
                x = ctanh(x)
            else:
                x = torch.tanh(x)

        x_fno = self.convs[index](true_dequantize(x), output_shape=output_shape)
        #self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = true_quantize(true_dequantize(x_fno) + true_dequantize(x_skip_fno))

        if (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        x = true_dequantize(self.channel_mlp[index](x)) + x_skip_channel_mlp

        if self.norm is not None:
            x = self.norm[self.n_norms * index + 1](x)

        if index < (self.n_layers - 1):
            x = self.non_linearity(x)

        return x

def quantised_forward_with_preactivation_docano(self, x, index=0, output_shape=None):
    # Apply non-linear activation (and norm)
    # before this block's convolution/forward pass:
    x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index](x)

    x_skip_fno = self.fno_skips[index](x)
    x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

    x_skip_channel_mlp = self.channel_mlp_skips[index](x)
    x_skip_channel_mlp = self.convs[index].transform(x_skip_channel_mlp, output_shape=output_shape)

    if self.stabilizer == "tanh":
        if self.complex_data:
            x = ctanh(x)
        else:
            x = torch.tanh(x)

    x_fno = self.convs[index](x, output_shape=output_shape)

    x = x_fno + x_skip_fno

    if index < (self.n_layers - 1):
        x = self.non_linearity(x)

    if self.norm is not None:
        x = self.norm[self.n_norms * index + 1](x)

    x = self.channel_mlp[index](x) + x_skip_channel_mlp

    return x

def quantised_compute_attention(self, tokens, batch_size):
        """
        Compute the key-query-value variant of the attention matrix for input token functions.

        Parameters
        ----------
        tokens : torch.Tensor
            Input tokens with shape (b * t, d, h, w, ...), where:
            b is the batch size,
            t is the number of tokens,
            d is the token codimension,
            and h, w, ... are the domain dimensions.
            Assumes input tokens have been normalized.
        
        batch_size : int
            The size of the batch.
        """

        k = self.Key(tokens)
        q = self.Query(tokens)
        v = self.Value(tokens)
        assert k.size(
            1) % self.n_heads == 0, "Number of channels in k, q, and v should be divisible by number of heads"

        # reshape from (b*t) (n*d) h w -> b n t (d*h*w ...)
        t = k.size(0) // batch_size  # Compute the number of tokens `t`
        # Computer per head token codimension `d`
        d = k.size(1) // self.n_heads

        # reshape from (b*t) (n*d) h w ... to b n t d h w ...
        k = k.view(batch_size, t, self.n_heads, d, *k.shape[-self.n_dim:])
        q = q.view(batch_size, t, self.n_heads, d, *q.shape[-self.n_dim:])
        v = v.view(batch_size, t, self.n_heads, d, *v.shape[-self.n_dim:])

        k = torch.transpose(k, 1, 2)
        q = torch.transpose(q, 1, 2)
        v = torch.transpose(v, 1, 2)
        # reshape
        k = true_dequantize(k.view(batch_size, self.n_heads, t, -1))
        q = true_dequantize(q.view(batch_size, self.n_heads, t, -1))
        v = true_dequantize(v.view(batch_size, self.n_heads, t, -1))

        # attention mechanism
        dprod = (torch.matmul(q, k.transpose(-1, -2)) /
                 (np.sqrt(k.shape[-1]) * self.temperature))
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)

        # Reshape from (b, n, t, d * h * w) to (b, n, t, d, h, w, ...)
        attention = attention.view(
            attention.size(0),
            attention.size(1),
            attention.size(2),
            d,
            *tokens.shape[-self.n_dim:])
        attention = torch.transpose(attention, 1, 2)
        attention = attention.reshape(attention.size(0) * attention.size(1),
                                      attention.size(2) * d,
                                      *tokens.shape[-self.n_dim:])

        return attention