import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from ..layers.channel_mlp import ChannelMLP
from ..layers.spectral_convolution import SpectralConv
from ..layers.skip_connections import skip_connection
from ..layers.padding import DomainPadding
from ..layers.coda_layer import CODALayer
from ..layers.resample import resample
from ..layers.embeddings import GridEmbedding2D, GridEmbeddingND
from .base_model import BaseModel
import types

class CODANO(BaseModel, name='CODANO'):
    _name = "CODANO"
    _version = "1.0"
    """Codomain Attention Neural Operators (CoDA-NO) uses a specialized attention mechanism in the codomain space for data in
    infinite dimensional spaces as described in [1]_. The model treates each input channel as a variable of the physical system
    and uses attention mechanism to model the interactions between the variables. The model uses lifting and projection modules
    to map the input variables to a higher-dimensional space and then back to the output space. The model also supports positional
    encoding and static channel information for additional context of the physical system such as external force or inlet condition.


    Parameters
    ----------
    output_variable_codimension : int
        The number of output channels (or output codomain dimension) corresponding to each input variable (or input channel). Default is 1.
        Example: For a input with 3 variables (channels) and output_variable_codimension=2, the output will have 6 channels (3 variables × 2 codimension).

    lifting_channels : int
        Number of intermidiate channels in the lifting block. The lifting module projects each input variable (i.e., each input channel) into a
        higher-dimensional space determied by `hidden_variable_codimension`. Default is 64 (two times the hidden_variable_codimension).
        If lifting_channels is None, lifting is not performed and the input channels are directly used as tokens for codoamin attention.

    hidden_variable_codimension : int
        The number of hidden channels corresponding to each input variable (or channel). Each input channel is independently lifted
        to `hidden_variable_codimension` channels by the lifting block. Default is 32.


    projection_channels : int. The number of intermidiate channels in the projection block of the codano is, default is 64. if `projection_channels=None`,
        projection is not performed and the output of the last CoDA block is returned directly.


    use_positional_encoding : bool
        Indicates whether to use variable-specific positional encoding. If True, a learnable positional encoding is concatenated
        to each variable (each input channel) before the lifting operation. The positinal encoding used here is a function space
        generalization of the learable positional encoding used in BERT [2]. In codano, the positional encoding is a function on
        domain which is learned directly in the Fourier Space. Default is False.

    positional_encoding_dim : int
        The dimension (number of channels) of the positional encoding learned of each input variable (i.e., input channel). Default is 8.

    positional_encoding_modes : list
        Number of Fourier modes used in positional encoding along each dimension. The positional embeddings are functions and are directly learned
        in Fourier space. This parameter must be specified when `use_positional_encoding=True`. Default is None.
        Example: For a 2D input, positional_encoding_modes could be [16, 16].

    static_channel_dim : int
        The number of channels for static information, such as boundary conditions in PDEs. These channels are concatenated with
        each variable before the lifting operation and use to provide additional information regarding the physical setup of the system.
        When `static_channel_dim > 0`, additional information must be provided during
        the forward pass. Default is 0.

        For example, static_channel_dim=1 can be used to provid mask of the domain pointing a hole or obstacle in the domain.

    variable_ids : list[str]
        The names of the variables in the dataset. Default is None.

        This parameter is **only** required when `use_positional_encoding=True` to initialize learnable positional embeddings for
        each unique physical varibles in the dataset.

        For example:
        If the dataset consists of only Navier Stokes equations, the variable_ids=['u_x', 'u_y', 'p'], representing the velocity
        components in x and y directions and pressure, respectively. Please note that we consider each input channel as a physical
        variable of the PDE.

        Please note that the 'velocity' variable is composed of two channels (codimension=2) and we have split the velocity field
        into two components, i.e., u_x and u_y. And this is to be done for all variable with codimension > 1.

        If the dataset consists of multiple PDEs, such as Navier Stokes and Heat equation, the variable_ids=['u_x', 'u_y', 'p', 'T'],
        where 'T' represents the temperature variable for thee Heat equation and 'u_x', 'u_y', 'p' are the velocity components and pressure
        for the Navier Stokes equations. This is required when we aim to learn a single solver for multiple different PDEs.

        This parameter is not required when `use_positional_encoding=False`.

    n_layers : int
        The number of codomain attention layers. Default is 4.

    n_modes : list
        The number of Fourier modes to use in integral operators in the CoDA-NO block along each dimension. Default is None.
        Example: For a 5-layer 2D CoDA-NO, n_modes=[[16, 16], [16, 16], [16, 16], [16, 16], [16, 16]].

    per_layer_scaling_factor : list
        The output scaling factor for each CoDANO_block along each dimension. The output of each of the CoDANO_block
        is resampled accroding to the scaling factor and then passed to the following CoDANO_blocks. Default is None ,i.e., no scaling.

        Example: For a 2D input and `n_layers=5`, per_layer_scaling_factor=[[1, 1], [0.5, 0.5], [1, 1], [2, 2], [1, 1]], which downsample the
        output of the second layer by a factor of 2 and upsample the output of the fourth layer by a factor of 2.

        The resolution of the output of the codano model is determined by the product of the scaling factors of all the layers.

    n_heads : list
        The number of attention heads for each layer. Default is None, i.e., single attention head for
        each codomain attention block.
        Example: For a 4-layer CoDA-NO, n_heads=[2, 2, 2, 2].

    attention_scaling_factors : list
        Scaling factors in the codomain attention mechanism to scale the key and query functions. These scaling factors are used to resample
        the key and query function before calculating the attention matrix. It does not have any effect on the value funnctions
        in the codoamin attention mechanism, i.e., it does not change the output shape of the block.  Default is None, which means no scaling.

        Example: For a 5-layer CoDA-NO, attention_scaling_factors=[0.5, 0.5, 0.5, 0.5, 0.5], which is downsample the key and query functions,
        reducing the resolution by a factor of 2.

    conv_module : nn.Module
        The convolution module to use in the CoDANO_block. Default is SpectralConv.

    nonlinear_attention : bool
        Indicates whether to use a non-linear attention mechanism, employing non-linear key, query, and value operators. Default is False.

    non_linearity : callable
        The non-linearity to use in the codomain attention block. Default is `F.gelu`.

    attention_token_dim : int
        The number of channels in each token function. `attention_token_dim` must divide `hidden_variable_codimension`. Default is 1.

    per_channel_attention : bool
        Indicates whether to use a per-channel attention mechanism in Codomain attention layer. Default is False.

    enable_cls_token : bool
        Indicates whether to use a learnable CLASS token during the attention mechanism. We use a function-space generalization of the
        learnable [class] token used in vision transformers such as ViT, which is learned directly in Fourier space. Default is False.

        The [class] function is realized on the input grid by performing an inverse Fourier transform of the learned Fourier coefficients.
        Then, the [class] token function is added to the set of input token functions before passing to the codomain attention layer. It aggregates
        information from all the other tokens through the attention mechanism. The output token corresponding to the [class] token is discarded in the
        output of the last CoDA block.

    Other parameters
    ----------------
    use_horizontal_skip_connection : bool, optional
        Indicates whether to use horizontal skip connections, similar to U-shaped architectures. Default is False.

    horizontal_skips_map : dict, optional
        A mapping that specifies horizontal skip connections between layers. Only required when `use_horizontal_skip_connection=True`. Default is None.
        Example: For a 5-layer architecture, horizontal_skips_map={4: 0, 3: 1} creates skip connections from layer 0 to layer 4 and layer 1 to layer 3.

    domain_padding : float
        The padding factor for each input channel. It zero pads each of the channel following the `domain_padding_mode`.  Default is 0.25.

    domain_padding_mode : str
        The domain padding mode, which can be 'one-sided' or 'two-sided'. Default is 'one-sided'.
        if one-sided, padding is only done along one side of each dimension.
        If two-sided, padding is done on both sides of each dimension.

    layer_kwargs : dict
        Additional arguments for the CoDA blocks. Default is an empty dictionary `{}`.

    References
    -----------
    .. [1] : Rahman, Md Ashiqur, et al. "Pretraining codomain attention neural operators for solving multiphysics pdes." (2024).
    NeurIPS 2024. https://arxiv.org/pdf/2403.12553.

    .. [2] : Devlin, Jacob, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.

    """

    def __init__(
        self,
        in_channels = 1,
        output_variable_codimension=1,
        lifting_channels: int = 64,
        hidden_variable_codimension=32,
        projection_channels: int = 64,
        use_positional_encoding=False,
        positional_encoding_dim=8,
        positional_encoding_modes=None,
        static_channel_dim=0,
        variable_ids=None,
        use_horizontal_skip_connection=False,
        horizontal_skips_map=None,
        n_layers=4,
        n_modes=None,
        per_layer_scaling_factors=None,
        n_heads=None,
        attention_scaling_factors=None,
        conv_module=SpectralConv,
        nonlinear_attention=False,
        non_linearity=F.gelu,
        attention_token_dim=1,
        per_channel_attention=False,
        layer_kwargs={},
        domain_padding=0.25,
        domain_padding_mode="one-sided",
        enable_cls_token=False,
    ):
        
        super().__init__()
        
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert (
            len(n_heads) == n_layers or n_heads is None
        ), "number of Attention head for all layers are not given"
        assert (
            len(per_layer_scaling_factors) == n_layers
            or per_layer_scaling_factors is None
        ), "scaling for all layers are not given"
        assert (
            len(attention_scaling_factors) == n_layers
            or attention_scaling_factors is None
        ), "attention scaling for all layers are not given"
        if use_positional_encoding:
            assert positional_encoding_dim > 0, "positional encoding dim is not given"
            assert (
                positional_encoding_modes is not None
            ), "positional encoding modes are not given"
        else:
            positional_encoding_dim = 0

        if attention_scaling_factors is None:
            attention_scaling_factors = [1] * n_layers

        input_variable_codimension = 1  # each channel is a variable
        if lifting_channels is None:
            self.lifting = False
        else:
            lifting_variable_codimension = lifting_channels
            self.lifting = True

        if projection_channels is None:
            self.projection = False
        else:
            projection_variable_codimension = projection_channels
            self.projection = True
        extended_variable_codimemsion = (
            input_variable_codimension + static_channel_dim + positional_encoding_dim
        )
        if not self.lifting:
            hidden_variable_codimension = extended_variable_codimemsion

        assert (
            hidden_variable_codimension % attention_token_dim == 0
        ), "attention token dim should divide hidden variable codimension"

        self.n_dim = len(n_modes[0])

        if n_heads is None:
            n_heads = [1] * n_layers
        if per_layer_scaling_factors is None:
            per_layer_scaling_factors = [[1] * self.n_dim] * n_layers
        if attention_scaling_factors is None:
            attention_scaling_factors = [1] * n_layers

        self.input_variable_codimension = input_variable_codimension
        self.hidden_variable_codimension = hidden_variable_codimension
        self.n_modes = n_modes
        self.per_layer_scale_factors = per_layer_scaling_factors
        self.non_linearity = non_linearity
        self.n_heads = n_heads
        self.enable_cls_token = enable_cls_token
        self.positional_encoding_dim = positional_encoding_dim
        self.variable_ids = variable_ids
        self.attention_scalings = attention_scaling_factors
        self.positional_encoding_modes = positional_encoding_modes
        self.static_channel_dim = static_channel_dim
        self.layer_kwargs = layer_kwargs
        self.use_positional_encoding = use_positional_encoding
        self.use_horizontal_skip_connection = use_horizontal_skip_connection
        self.horizontal_skips_map = horizontal_skips_map
        self.output_variable_codimension = output_variable_codimension

        if self.positional_encoding_modes is not None:
            self.positional_encoding_modes[-1] = self.positional_encoding_modes[-1] // 2

        # calculating scaling
        if self.per_layer_scale_factors is not None:
            self.end_to_end_scaling = [1] * len(self.per_layer_scale_factors[0])
            # multiplying scaling factors
            for k in self.per_layer_scale_factors:
                self.end_to_end_scaling = [
                    i * j for (i, j) in zip(self.end_to_end_scaling, k)
                ]
        else:
            self.end_to_end_scaling = [1] * self.n_dim

        if self.n_heads is None:
            self.n_heads = [1] * self.n_layers

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                resolution_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.extended_variable_codimemsion = extended_variable_codimemsion
        if self.lifting:
            self.lifting = ChannelMLP(
                in_channels=extended_variable_codimemsion,
                out_channels=self.hidden_variable_codimension,
                hidden_channels=lifting_variable_codimension,
                n_layers=2,
                n_dim=self.n_dim,
            )
        else:
            self.hidden_variable_codimension = self.extended_variable_codimemsion

        self.attention_layers = nn.ModuleList([])

        for i in range(self.n_layers):
            self.attention_layers.append(
                CODALayer(
                    n_modes=self.n_modes[i],
                    n_heads=self.n_heads[i],
                    scale=self.attention_scalings[i],
                    token_codimension=attention_token_dim,
                    per_channel_attention=per_channel_attention,
                    nonlinear_attention=nonlinear_attention,
                    resolution_scaling_factor=self.per_layer_scale_factors[i],
                    conv_module=conv_module,
                    non_linearity=self.non_linearity,
                    **self.layer_kwargs,
                )
            )

        if self.use_horizontal_skip_connection:
            # horizontal skip connections
            # linear projection of the concated tokens from skip connections

            self.skip_map_module = nn.ModuleDict()
            for k in self.horizontal_skips_map.keys():
                self.skip_map_module[str(k)] = ChannelMLP(
                    in_channels=2 * self.hidden_variable_codimension,
                    out_channels=self.hidden_variable_codimension,
                    hidden_channels=None,
                    n_layers=1,
                    non_linearity=nn.Identity(),
                    n_dim=self.n_dim,
                )

        if self.projection:
            self.projection = ChannelMLP(
                in_channels=self.hidden_variable_codimension,
                out_channels=output_variable_codimension,
                hidden_channels=projection_variable_codimension,
                n_layers=2,
                n_dim=self.n_dim,
            )
        else:
            self.projection = None

        if enable_cls_token:
            self.cls_token = nn.Parameter(
                torch.randn(
                    1,
                    self.hidden_variable_codimension,
                    *self.n_modes[0],
                    dtype=torch.cfloat,
                )
            )

        if use_positional_encoding:
            self.positional_encoding = nn.ParameterDict()
            for i in self.variable_ids:
                self.positional_encoding[i] = nn.Parameter(
                    torch.randn(
                        1,
                        positional_encoding_dim,
                        *self.positional_encoding_modes,
                        dtype=torch.cfloat,
                    )
                )

    def _extend_positional_encoding(self, new_var_ids):
        """
        Add variable specific positional encoding for new variables. This function is required
        while adapting a pre-trained model to a new dataset/PDE with additional new variables.

        Parameters
        ----------
        new_var_ids : list[str]
            IDs of the new variables to add positional encoding.
        """
        for i in range(new_var_ids):
            self.positional_encoding[i] = nn.Parameter(
                torch.randn(
                    1,
                    self.positional_encoding_dim,
                    *self.positional_encoding_modes,
                    dtype=torch.cfloat,
                )
            )

        self.variable_ids += new_var_ids

    def _get_positional_encoding(self, x, input_variable_ids):
        """
        Returns the positional encoding for the input variables.
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
        """
        encoding_list = []
        for i in input_variable_ids:
            encoding_list.append(
                torch.fft.irfftn(self.positional_encoding[i], s=x.shape[-self.n_dim :])
            )

        return torch.stack(encoding_list, dim=1)

    def _get_cls_token(self, x):
        """
        Returns the learnable cls token for the input variables.
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
            This is used to determine the shape of the cls token.
        """
        cls_token = torch.fft.irfftn(self.cls_token, s=x.shape[-self.n_dim :])
        repeat_shape = [1 for _ in x.shape]
        repeat_shape[0] = x.shape[0]
        cls_token = cls_token.repeat(*repeat_shape)
        return cls_token

    def _extend_variables(self, x, static_channel, input_variable_ids):
        """
        Extend the input variables by concatenating the static channel and positional encoding.
        Parameters
        ----------
        x : torch.Tensor
            input tensor of shape (batch_size, num_inp_var, H, W, ...)
        static_channel : torch.Tensor
            static channel tensor of shape (batch_size, static_channel_dim, H, W, ...)
        input_variable_ids : list[str]
            The names of the variables corresponding to the channels of input 'x'.
        """
        x = x.unsqueeze(2)
        if static_channel is not None:
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[1] = x.shape[1]
            static_channel = static_channel.unsqueeze(1).repeat(*repeat_shape)
            x = torch.cat([x, static_channel], dim=2)
        if self.use_positional_encoding:
            positional_encoding = self._get_positional_encoding(x, input_variable_ids)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat([x, positional_encoding.repeat(*repeat_shape)], dim=2)
        return x

    def forward(self, **kwargs):
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
            x = self.lifting(x)
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
                x = self.skip_map_module[str(layer_idx)](x)
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
            x = self.projection(x)
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

    def save_model(self, destination=None, prefix='', keep_vars=False):
        """Create a state dictionary with model parameters and special components."""
        # Create state dict directly from named parameters
        state_dict = {
            prefix + key: val.detach() if not keep_vars else val
            for key, val in self.named_parameters()
        }
        
        # Add buffers (like running stats in BatchNorm)
        state_dict.update({
            prefix + key: val.detach() if not keep_vars else val
            for key, val in self.named_buffers()
        })
        
        # Handle positional encoding separately if it exists
        if hasattr(self, 'positional_encoding') and isinstance(self.positional_encoding, nn.ParameterDict):
            for key, value in self.positional_encoding.items():
                state_dict[prefix + f'positional_encoding.{key}'] = value if keep_vars else value.detach()
        
        # Store non_linearity function name if it's a standard PyTorch function
        if hasattr(self, 'non_linearity'):
            if self.non_linearity == F.gelu:
                state_dict[prefix + 'non_linearity'] = 'gelu'
            # Add other function mappings as needed
        
        return state_dict

    def load_model(self, state_dict, strict=True):
        """Load a state dictionary into the model."""
        missing_keys = []
        unexpected_keys = []
        
        # Handle regular parameters and buffers
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].clone()
            elif strict:
                missing_keys.append(name)
                
        for name, buf in self.named_buffers():
            if name in state_dict:
                buf.data = state_dict[name].clone()
            elif strict:
                missing_keys.append(name)
        
        # Handle positional encoding
        if hasattr(self, 'positional_encoding'):
            self.positional_encoding = nn.ParameterDict()
            for key in list(state_dict.keys()):
                if key.startswith('positional_encoding.'):
                    param_key = key.split('.')[1]
                    self.positional_encoding[param_key] = nn.Parameter(state_dict[key].clone())
        
        # Handle non_linearity function
        if 'non_linearity' in state_dict:
            func_name = state_dict['non_linearity']
            if func_name == 'gelu':
                self.non_linearity = F.gelu
            # Add other function mappings as needed
        
        # Check for unexpected keys
        if strict:
            for key in state_dict.keys():
                if not any(key.startswith(prefix) for prefix in ['positional_encoding.', 'non_linearity']):
                    if key not in dict(self.named_parameters()) and key not in dict(self.named_buffers()):
                        unexpected_keys.append(key)
        
        if strict and (missing_keys or unexpected_keys):
            error_msg = []
            if missing_keys:
                error_msg.append(f'Missing key(s) in state_dict: {", ".join(missing_keys)}')
            if unexpected_keys:
                error_msg.append(f'Unexpected key(s) in state_dict: {", ".join(unexpected_keys)}')
            raise RuntimeError('\n'.join(error_msg))
        
        return self



# define the foundation codano here

from functools import partial
import logging
from typing import Literal, NamedTuple, Optional
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from neuralop.layers.padding import DomainPadding
from neuralop.layers.codano_block_nd import CodanoBlockND
from neuralop.layers.fino_nd import SpectralConvKernel2d
from neuralop.layers.variable_encoding import VariableEncoding2d
# TODO replace with nerualop.MLP module
class PermEqProjection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_dim=2,
        non_linearity=F.gelu,
        permutation_invariant=False,
    ):
        """Permutation invariant projection layer.

        Performs linear projections on each channel separately.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (in_channels
                                if hidden_channels is None else
                                hidden_channels)
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')

        self.permutation_invariant = permutation_invariant

        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.norm = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        batch = x.shape[0]
        if self.permutation_invariant:
            assert x.shape[1] % self.in_channels == 0, \
                "Total Number of Channels is not divisible by number of tokens"
            x = rearrange(x, 'b (g c) h w -> (b g) c h w', c=self.in_channels)

        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        if self.permutation_invariant:
            x = rearrange(x, '(b g) c h w -> b (g c) h w', b=batch)
        return x


class VariableEncodingArgs(NamedTuple):
    basis: Literal["sht", "fft"]
    n_channels: int
    """Number of extra encoding channels per variable."""
    modes_x: int
    modes_y: int
    modes_t: Optional[int] = None


class CodANO(nn.Module):
    """
    Parameters
    ---
    input_token_codimension : input token codim/number of channel per input token
    out_token_codim=None : output token codim/number of channel per output token
    hidden_token_codim=None :
    lifting_token_codim=None :
    var_encoding=False : boolean
        if true then it adds variable encoding with each channel
    var_num=None :  denotes the number of variables
    var_enco_basis='sht' :  specify the basis funtion for variable encodings
    var_enco_channels=1 : number of channels for each variable encoding
    var_enco_mode_x=50 : number of x modes for each variable encoding
    var_enco_mode_y=50 : number of y models for each variable encoding
    enable_cls_token=False : if true, learnable cls token will be added
    static_channels_num=0 :
        Number of static channels to be concatenated (xy grid, land/sea mask etc)
    static_features=None :
        The static feature (it will be taken from the Preprocessor while
        initializing the model)
    integral_operator_top :
        Required for the re-grid operation (for example: from equiangular to LG grid.)
    integral_operator_bottom :
        Required for the re-grid operation (for example: from LG grid to equiangular)
    """

    def __init__(
        self,
        input_token_codimension,
        output_token_codimension=None,
        hidden_token_codimension=None,
        lifting_token_codimension=None,
        n_layers=4,
        n_modes=None,
        max_n_modes=None,
        scalings=None,
        n_heads=1,
        non_linearity=F.gelu,
        layer_kwargs={'use_mlp': False,
                      'mlp_dropout': 0,
                      'mlp_expansion': 1.0,
                      'non_linearity': F.gelu,
                      'norm': None,
                      'preactivation': False,
                      'fno_skip': 'linear',
                      'horizontal_skip': 'linear',
                      'mlp_skip': 'linear',
                      'separable': False,
                      'factorization': None,
                      'rank': 1.0,
                      'fft_norm': 'forward',
                      'normalizer': 'instance_norm',
                      'joint_factorization': False,
                      'fixed_rank_modes': False,
                      'implementation': 'factorized',
                      'decomposition_kwargs': dict(),
                      'normalizer': False},
        per_channel_attention=True,
        operator_block=CodanoBlockND,
        integral_operator=SpectralConvKernel2d,
        integral_operator_top=partial(
            SpectralConvKernel2d, sht_grid="legendre-gauss"),
        integral_operator_bottom=partial(
            SpectralConvKernel2d, isht_grid="legendre-gauss"),
        projection=True,
        lifting=True,
        domain_padding=0.5,
        domain_padding_mode='one-sided',
        n_variables=None,
        variable_encoding_args: VariableEncodingArgs = None,
        enable_cls_token=False,
        logger=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(
            n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, \
            "number of Attention head for all layers are not given"
        if integral_operator_bottom is None:
            integral_operator_bottom = integral_operator
        if integral_operator_top is None:
            integral_operator_top = integral_operator
        self.n_dim = len(n_modes[0])
        self.input_token_codimension = input_token_codimension
        # self.n_variables = n_variables
        if hidden_token_codimension is None:
            hidden_token_codimension = input_token_codimension
        if lifting_token_codimension is None:
            lifting_token_codimension = input_token_codimension
        if output_token_codimension is None:
            output_token_codimension = input_token_codimension

        self.hidden_token_codimension = hidden_token_codimension
        self.n_modes = n_modes
        self.max_n_modes = max_n_modes
        self.scalings = scalings
        self.non_linearity = non_linearity
        self.n_heads = n_heads
        self.integral_operator = integral_operator
        self.lifting = lifting
        self.projection = projection
        self.num_dims = len(n_modes[0])
        self.enable_cls_token = enable_cls_token

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.layer_kwargs = layer_kwargs
        if layer_kwargs is None:
            self.layer_kwargs = {
                'incremental_n_modes': None,
                'use_mlp': False,
                'mlp_dropout': 0,
                'mlp_expansion': 1.0,
                'non_linearity': F.gelu,
                'norm': None,
                'preactivation': False,
                'fno_skip': 'linear',
                'horizontal_skip': 'linear',
                'mlp_skip': 'linear',
                'separable': False,
                'factorization': None,
                'rank': 1.0,
                'fft_norm': 'forward',
                'normalizer': 'instance_norm',
                'joint_factorization': False,
                'fixed_rank_modes': False,
                'implementation': 'factorized',
                'decomposition_kwargs': None,
            }

        # self.n_static_channels = n_static_channels
        """The number of static channels for all variable channels."""

        # calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling = self.get_output_scaling_factor(
                np.ones_like(self.scalings[0]),
                self.scalings
            )
        else:
            self.end_to_end_scaling = 1
        self.logger.debug(f"{self.end_to_end_scaling=}")
        if isinstance(self.end_to_end_scaling, (float, int)):
            self.end_to_end_scaling = [self.end_to_end_scaling] * self.n_dim

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        # A variable + it's variable encoding + the static channel(s)
        # together constitute a token
        # n_lifted_channels = self.input_token_codimension + \
        #     variable_encoding_args.n_channels + \
        #     self.n_static_channels
        if self.lifting:
            self.lifting = PermEqProjection(
                in_channels=input_token_codimension,
                out_channels=hidden_token_codimension,
                hidden_channels=lifting_token_codimension,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
                permutation_invariant=True,   # Permutation
            )
        # elif self.use_variable_encoding:
        #     hidden_token_codimension = n_lifted_channels

        cls_dimension = 1 if enable_cls_token else 0
        self.codimension_size = hidden_token_codimension * n_variables + cls_dimension

        self.logger.debug(
            f"Expected number of channels: {self.codimension_size=}")

        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0 and self.n_layers != 1:
                conv_op = integral_operator_top
            elif i == self.n_layers - 1 and self.n_layers != 1:
                conv_op = integral_operator_bottom
            else:
                conv_op = self.integral_operator

            self.base.append(
                operator_block(
                    n_modes=self.n_modes[i],
                    max_n_modes=self.max_n_modes[i],
                    n_head=self.n_heads[i],
                    token_codim=hidden_token_codimension,
                    output_scaling_factor=[self.scalings[i]],
                    SpectralConvolution=conv_op,
                    codim_size=self.codimension_size,
                    per_channel_attention=per_channel_attention,
                    num_dims=self.num_dims,
                    logger=self.logger.getChild(f"base[{i}]"),
                    **self.layer_kwargs,
                )
            )

        if self.projection:
            self.projection = PermEqProjection(
                in_channels=hidden_token_codimension,
                out_channels=output_token_codimension,
                hidden_channels=lifting_token_codimension,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
                permutation_invariant=True,   # Permutation
            )

        if enable_cls_token:
            self.cls_token = VariableEncoding2d(
                1,
                hidden_token_codimension,
                (variable_encoding_args.modes_x,
                 variable_encoding_args.modes_y),
                basis=variable_encoding_args.basis)

    def get_output_scaling_factor(self, initial_scale, scalings_per_layer):
        for k in scalings_per_layer:
            initial_scale = np.multiply(initial_scale, k)
        initial_scale = initial_scale.tolist()
        if len(initial_scale) == 1:
            initial_scale = initial_scale[0]
        return initial_scale

    def get_device(self,):
        return self.cls_token.coefficients_r.device

    def forward(self, x: torch.Tensor):
        if self.lifting:
            x = self.lifting(x)

        if self.enable_cls_token:
            cls_token = self.cls_token(x).unsqueeze(0)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat(
                [
                    cls_token.repeat(*repeat_shape),
                    x,
                ],
                dim=1,
            )

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape_en = [round(i * j) for (i,
                                             j) in zip(x.shape[-self.n_dim:],
                                                       self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):
            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape=cur_output_shape)
            # self.logger.debug(f"{x.shape} (block[{layer_idx}])")

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.projection:
            x = self.projection(x)
            # self.logger.debug(f"{x.shape} (projection)")

        return x


class CoDANOTemporal:
    def __call__(self, x):
        pass