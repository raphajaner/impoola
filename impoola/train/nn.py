import numpy as np
import torch
import torch.nn as nn
from impoola.train.moe import SoftMoE, ExpertModel
import torch.nn.functional as F


def layer_init_kaiming_uniform(layer, a=0, nonlinearity='relu'):
    nn.init.kaiming_uniform_(layer.weight, a=a, nonlinearity=nonlinearity)
    # nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    nn.init.constant_(layer.bias, 0)
    return layer


def layer_init_conv_torch_standard(layer):
    nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        if fan_in != 0:
            bound = 1 / np.sqrt(fan_in)
            layer.uniform_(layer.bias, -bound, bound)
    return layer


def layer_init_xavier_uniform(layer, gain=1.0):
    nn.init.xavier_uniform_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)
    return layer


def layer_init_orthogonal(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def activation_factory(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'rrelu':
        return nn.RReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation == 'silu':
        return nn.SiLU()
    else:
        raise NotImplementedError


def encoder_factory(encoder_type, use_moe, *args, **kwargs):
    if encoder_type == 'impala':
        model = ImpalaCNN(*args, **kwargs)
        if use_moe:
            # remove the linear head and only keep the CNN backbone
            backbone = model.network[:-4]
            assert backbone[-1].__class__ == ConvSequence

            # import pdb; pdb.set_trace()

            h = w = 1 if kwargs['use_pooling_layer'] else 8
            token_length = kwargs['width_scale'] * kwargs['cnn_filters'][-1]
            num_experts = 10
            expert_hidden_size = kwargs['out_features']

            model = SoftMoE(
                module=ExpertModel,
                backbone=backbone,
                num_experts=num_experts, num_tokens=h * w, token_length=token_length,
                expert_hidden_size=expert_hidden_size,
                capacity_factor=1, expert_type="SMALL", normalization=False, use_random_phi=False
            )
            out_features = expert_hidden_size * h * w
        else:
            out_features = kwargs['out_features']
        return model, out_features

    elif encoder_type == 'nature':
        return NatureCNN(*args, **kwargs), kwargs['out_features']
    else:
        raise NotImplementedError(f"Unsupported encoder type: {encoder_type}")


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale, use_layer_init_normed=False, activation='relu'):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        # scale = np.sqrt(scale)
        kernel_size = 3
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        # self.conv0 = layer_init_kaiming_uniform(conv0)
        self.conv0 = conv0
        # self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale) if use_layer_init_normed else conv0

        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same')
        # self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale) if use_layer_init_normed else conv1
        # self.conv1 = layer_init_kaiming_uniform(conv1)
        self.conv1 = conv1

        self.activation0 = activation_factory(activation)
        self.activation1 = activation_factory(activation)

    def forward(self, x):
        inputs = x
        x = self.activation0(x)
        x = self.conv0(x)
        x = self.activation1(x)
        x = self.conv1(x)
        return x + inputs


class LinearResidualBlock(nn.Module):
    # Based on https://github.com/SonyResearch/simba/blob/master/scale_rl/networks/layers.py
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.float32):
        super(LinearResidualBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # self.layer_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)

        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        # x = self.layer_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return res + x


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale, use_layer_init_normed=False, activation='relu',
                 positional_encoding=None):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels

        # Input convolution and pooling
        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                         padding="same")
        # self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0) if use_layer_init_normed else conv
        # self.conv = layer_init_kaiming_uniform(conv) #, nonlinearity="linear")
        self.conv = conv

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        if positional_encoding == "after_max_pool":
            self.pooling = nn.Sequential(self.pooling,
                                         PositionalEncoding(input_shape[1] // 2, input_shape[2] // 2, out_channels))
        elif positional_encoding == "before_max_pool":
            self.pooling = nn.Sequential(PositionalEncoding(input_shape[1], input_shape[2], out_channels), self.pooling)

        # Residual blocks
        nblocks = 2
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale, use_layer_init_normed=use_layer_init_normed,
                                        activation=activation)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        # assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return self._out_channels, (h + 1) // 2, (w + 1) // 2


class GlobalSumPool2d(nn.Module):
    def forward(self, x):
        return x.sum(dim=(2, 3))


class GlobalFeatureAvgPool2d(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], x.shape[1], -1).mean(dim=1)


class ImpalaCNN(nn.Module):
    def __init__(
            self, envs,
            width_scale=1, out_features=256, cnn_filters=(16, 32, 32), activation='relu',
            use_layer_init_normed=False,
            use_pooling_layer=False, pooling_layer_kernel_size=1,
            use_dropout=False,
            use_1d_conv=False,
            use_depthwise_conv=False,
            use_simba=False,
            positional_encoding=None
    ):
        super().__init__()
        self.use_simba = use_simba
        self.simba_scale = 3

        shape = envs.single_observation_space.shape  # (c, h, w)
        scale = 1 / np.sqrt(len(cnn_filters))  # Not fully sure about the logic behind this but it's used in PPG code

        # CNN backbone
        cnn_layers = []

        for i_block, out_channels in enumerate(cnn_filters):
            if positional_encoding == "last":
                if i_block + 1 == len(cnn_filters):
                    cnn_layers += [PositionalEncoding(shape[1], shape[2], shape[0])]
            elif positional_encoding == "first":
                if i_block == 0:
                    cnn_layers += [PositionalEncoding(shape[1], shape[2], shape[0])]
            elif positional_encoding == "all":
                cnn_layers += [PositionalEncoding(shape[1], shape[2], shape[0])]

            conv_seq = ConvSequence(shape, int(out_channels * width_scale), scale=scale,
                                    use_layer_init_normed=use_layer_init_normed, activation=activation,
                                    positional_encoding=positional_encoding)

            shape = conv_seq.get_output_shape()
            cnn_layers.append(conv_seq)

        # TODO: Breaking change to before (Jan 24)
        cnn_layers += [activation_factory(activation)]

        # ImpoolaCNN improves the original IMPALA CNN by adding a pooling layer
        if use_pooling_layer:
            if pooling_layer_kernel_size == -1:
                # use the stacked approach where gap and (2, 2) pooling are used at the same time
                cnn_layers += [
                    StackedAdaptiveAvgPool2d()
                ]
            else:
                cnn_layers += [
                    nn.AdaptiveAvgPool2d((pooling_layer_kernel_size, pooling_layer_kernel_size))

                    # do sum pooling instead of avg pooling
                    # GlobalSumPool2d()
                    # GlobalFeatureAvgPool2d()

                    # TODO!!!! JUST FOR TESTING!!!!!!!!!!!!!
                    # nn.AdaptiveMaxPool2d((pooling_layer_kernel_size, pooling_layer_kernel_size))
                    # given the output shape, calculate an average pooling layer that reduces to 2x2 with some overlap
                    # nn.AvgPool2d(kernel_size=5, stride=4, padding=2)
                    # nn.AvgPool2d(kernel_size=6, stride=2, padding=0)
                ]

        if use_1d_conv:
            cnn_layers += [
                nn.Conv1d(in_channels=shape[0], out_channels=shape[1] * shape[2], kernel_size=3, padding=1)
            ]

        if use_depthwise_conv:
            cnn_layers += [
                nn.Conv2d(in_channels=shape[0], out_channels=shape[0], kernel_size=shape[1], groups=shape[0], padding=0)
            ]

        # Linear head
        linear_layers = cnn_layers
        linear_layers += [nn.Flatten()]

        if use_dropout:
            linear_layers += [nn.Dropout(0.1)]

        if self.use_simba:
            for _ in range(self.simba_scale):
                linear_layers += [LinearResidualBlock(shape[0])]
            # add layer norm at the end
            # linear_layers += [nn.LayerNorm(shape[0])]
            linear_layers += [activation_factory(activation)]
        else:
            # encodertop = nn.LazyLinear(out_features)  # in_features=shape[0] * shape[1] * shape[2]
            if use_pooling_layer or use_1d_conv or use_depthwise_conv:
                if pooling_layer_kernel_size == -1:
                    in_features_encoder = shape[0] * 2 * 2 + shape[0] * 1 * 1
                else:
                    in_features_encoder = shape[0] * pooling_layer_kernel_size * pooling_layer_kernel_size
            else:
                in_features_encoder = shape[0] * shape[1] * shape[2]
            encodertop = nn.Linear(in_features_encoder, out_features=out_features)

            # encodertop = layer_init_kaiming_uniform(encodertop)  # TODO: Orthogonal could be better

            # encodertop = nn.LazyLinear(out_features * 2)  # in_features=shape[0] * shape[1] * shape[2]
            # encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4) if use_layer_init_normed else encodertop

            linear_layers += [
                # activation_factory(activation),
                encodertop,
                activation_factory(activation)
            ]
        self.network = nn.Sequential(*linear_layers)

    def forward(self, x):
        # add a positional signal as overlay to the input image
        # x = x + self.positional_signal(x)
        x = x / 255.0
        return self.network(x)

    def get_output_shape(self):
        return self.network[-2].out_features

    # def positional_signal(self, x):
    #     # add a positional signal as overlay to the input image using a sin cos signal
    #     b, c, h, w = x.shape
    #     assert h == w
    #     pos = torch.arange(h, device=x.device).float()
    #     pos = pos / h
    #     pos = pos.unsqueeze(0).unsqueeze(0).expand(b, 1, h)
    #     pos = pos.unsqueeze(2).expand(b, 1, h, w)
    #     pos = torch.cat([torch.sin(pos * np.pi), torch.cos(pos * np.pi)], dim=1)
    #     return pos

    # class PositionalEncoding(nn.Module):
    #     def __init__(self, height, width, d_model):
    #         super(PositionalEncoding, self).__init__()
    #
    #         # Create the positional encoding matrix
    #         pe = torch.zeros(d_model, height, width)
    #         y_pos, x_pos = torch.meshgrid(torch.arange(height), torch.arange(width))
    #         position = torch.stack([x_pos, y_pos], dim=0)  # Shape: [2, height, width]
    #
    #         # Calculate sinusoids for the positional encoding
    #         for i in range(0, d_model):
    #             if i % 2 == 0:
    #                 pe[i] = torch.sin(position[0] / (10000 ** (i / d_model)))  # sin for x position
    #             else:
    #                 pe[i] = torch.cos(position[1] / (10000 ** (i / d_model)))  # cos for y position
    #
    #         pe = pe.unsqueeze(0)  # Add batch dimension: [1, d_model, height, width]
    #         # add it as a buffer so it's saved in the state_dict
    #         # self.pe = nn.Parameter(pe, requires_grad=False)
    #         self.register_buffer('pe', pe)
    #
    #     def forward(self, x):
    #         return x + self.pe


class StackedAdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling1 = nn.Sequential(nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.pooling2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, x):
        x1 = self.pooling1(x)
        x2 = self.pooling2(x)
        return torch.cat([x1, x2], dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, height, width, d_model, scale=0.1):
        super().__init__()

        # Create positional grid
        y_pos, x_pos = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        position = torch.stack([x_pos, y_pos], dim=0).float()  # Shape: [2, height, width]

        # Initialize PE matrix
        pe = torch.zeros(d_model, height, width)

        # Compute sin/cos encoding
        for i in range(0, d_model, 2):
            div_term = 10000 ** (i / d_model)
            pe[i] = torch.sin(position[0] / div_term)  # Sin for x
            if i + 1 < d_model:
                pe[i + 1] = torch.cos(position[1] / div_term)  # Cos for y

        pe = pe.unsqueeze(0)  # Shape: [1, d_model, height, width]

        # Move pe to [0, 1] range and scale
        pe = (pe + 1) / 2 * scale
        # Register as buffer (non-trainable parameter)
        self.register_buffer('pe', pe)
        # self.alpha = nn.Parameter(torch.tensor(0.05))  # Small trainable factor

    def forward(self, x):
        return x + self.pe  # * self.alpha


class NatureCNN(nn.Module):
    def __init__(self, envs, width_scale=1, out_features=256, cnn_filters=(32, 64, 64), activation='relu',
                 use_layer_init_normed=False,
                 use_pooling_layer=False, pooling_layer_kernel_size=1,
                 use_dropout=False,
                 use_1d_conv=False,
                 use_depthwise_conv=False,
                 use_simba=False,
                 positional_encoding=None
                 ):
        super().__init__()

        shape = envs.single_observation_space.shape  # (c, h, w)

        layers = [
            nn.Conv2d(shape[0], cnn_filters[0] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[0] * width_scale, cnn_filters[1] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[1] * width_scale, cnn_filters[2] * width_scale, 4, stride=2),
            activation_factory(activation),
            nn.Conv2d(cnn_filters[3] * width_scale, cnn_filters[3] * width_scale, 3, stride=1),
            activation_factory(activation)
        ]

        if use_pooling_layer:
            layers = layers[:-1]
            layers += [
                nn.AdaptiveAvgPool2d((pooling_layer_kernel_size, pooling_layer_kernel_size)),
                activation_factory(activation)  # TODO: Check order of activation
            ]

        if use_dropout or use_1d_conv or use_depthwise_conv:
            raise NotImplementedError

        layers += [
            nn.Flatten(),
            nn.LazyLinear(out_features),  # layer_init(nn.Linear(64 * 7 * 7, 512)),  # TODO: Check activation!
            activation_factory(activation)
        ]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x / 255.0)
