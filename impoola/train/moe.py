import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Literal
import torch.nn.functional as F


class ExpertModel(nn.Module):
    """Neural network class for the expert networks of a Soft Mixture of Experts layer."""

    def __init__(
            self,
            in_features,  # better readability than token size, even though same value
            expert_hidden_size,
            maintain_token_size: bool = True,
            noisy: bool = False,
            # initializer=torch.nn.init.xavier_uniform_,
    ):
        """
        Args:
            in_features: Number of input features for the expert network. Equal to the token size.
            expert_hidden_size: Number of outputs units of the first dense layer of the expert network. A second dense
                                layer might be used if maintain_token_size=True, making this layer hidden.
            maintain_token_size: Whether to add a second dense layer to the expert network, which features as many
                                 neurons as the token size.
            initializer: Weight initialization type for the dense layers of the expert network.
        """
        super().__init__()
        self.in_features = in_features
        self.expert_hidden_size = expert_hidden_size
        self.maintain_token_size = maintain_token_size
        self.noisy = noisy
        # self.initializer = initializer

        if self.noisy:
            # TODO: Add noisy networks.
            pass
        else:
            self.net = nn.Linear(in_features=self.in_features, out_features=self.expert_hidden_size)

            if self.maintain_token_size:
                self.maintain_token_size_layer = nn.Linear(in_features=self.expert_hidden_size,
                                                           out_features=self.in_features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:  Tensor of shape [num_tokens, token_length]
        Returns:
            y: Tensor of shape [num_tokens, self.expert_hidden_size] if maintain_token_size=False and shape
            [num_tokens, token_length] if maintain_token_size=False.
        """

        if self.noisy:
            pass
        else:
            x = self.net(x)
        hidden_y = x
        y = F.relu(x)

        if self.maintain_token_size:
            y = self.maintain_token_size_layer(x)

        return y, hidden_y


def l2_normalize(x, dim, eps=1e-6):
    """
    Normalizes the input tensor along given dimension.

    Args:
        x: Input tensor
        dim: dimension along which to normalize
        eps: small value added to avoid division by zero
    Returns:
        Normalized tensor

    Jax code:
    def l2_normalize(x, axis, eps=1e-6):
        norm = jnp.sqrt(jnp.square(x).sum(axis=axis, keepdims=True))
        return x * jnp.reciprocal(norm + eps)
    """
    norm = torch.sqrt(torch.square(x).sum(dim=dim, keepdim=True))
    return x * torch.reciprocal(norm + eps)


class SoftMoE(nn.Module):
    """
    Implements a Soft Mixture of Experts layer (https://arxiv.org/pdf/2308.00951), translating the DRL implementation by
    Obando-Ceron at al. (https://arxiv.org/pdf/2402.08609) from Jax to PyTorch.
    """

    def __init__(self,
                 module,
                 backbone: nn.Module,
                 num_experts: int,
                 num_tokens: int,  # Not a constructor argument in the Ceron implementation (inferred from the data)
                 token_length: int,  # Not a constructor argument in the Ceron implementation (inferred from the data)
                 expert_hidden_size: int,
                 capacity_factor: float = 1.0,
                 expert_type: Literal['SMALL', 'BIG'] = 'SMALL',
                 normalization: bool = False,
                 use_random_phi: bool = False) -> None:
        """
        Parameters:
            module: Uninstantiated expert network class, later used to instantiate the different expert networks.
                    Imported from networks.py.
            num_experts (int): Number of expert networks used.
            num_tokens (int): Number of tokens for input and output of the Soft Mixture of Experts layer. For the
                              PerConv tokenization type, which is recommended by Obando-Ceron et al. and used in this
                              code, the number of tokens is equal to height*width of the incoming feature maps.
                              num_tokens is required to compute the number of slots based on the capacity factor and the
                              number of experts.
            token_length (int): Length/dimension of the tokens for input and output of the Soft Mixture of Experts
                                layer. For the PerConv tokenization type, which is recommended by Obando-Ceron et al.
                                and used in this code, the number of tokens is equal to the number of incoming feature
                                maps. token_length is needed to determine the dimensions of multiple weight matrices.
            expert_hidden_size (int): Number of neurons in the first layer of the expert network. If a second layer is
                                      used in the expert network, this layer becomes a hidden layer.
            capacity_factor (float): Higher values allocate more slots, i.e. capacity, to each expert. A capacity factor
                                     of 1 means that input tokens are equally distributed among the expert networks. For
                                     capacity_factor > 1, there are more slots than input tokens.
            expert_type: The original Ceron implementation offers the option to choose between a small expert network
                         with up to two dense layers and a large expert network, which can include more complex
                         architectural components such as IMPALA.
            normalization (bool): Normalization to maintain compatibility with layer normalization. Normalizes the input
                                  tensor along the token_length dimension dim=1 and self.phi_weights along dim=0 to
                                  prevent the softmax operation to collapse to a one-hot vector when layer normalization
                                  is used and the token length is large. See Appendix E in
                                  https://arxiv.org/pdf/2308.00951 for further details.
            use_random_phi (bool): If True, the per-slot parameters self.phi_weights are randomly initialized and not
                                   learnable. Obando-Ceron et al. use this experiment setup to demonstrate that the
                                   learnable phi_weights are an important contributor to model performance and
                                   distributing tokens to experts is not the sole cause for the performance improvements
                                   observed.
        """
        super().__init__()

        # initialize instance attributes for constructor arguments
        self.module = module
        self.backbone = backbone
        self.num_experts = num_experts
        self.num_tokens = num_tokens
        self.token_length = token_length
        self.capacity_factor = capacity_factor
        # kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
        self.expert_type = expert_type
        self.normalization = normalization
        self.use_random_phi = use_random_phi
        self.fast_forward = True

        # Ceron: Create phi weight matrix of size (d x n.p), where d is the token dim,
        # n is the number of experts and p is the capacity of each expert (#slots).
        # Ceron: TODO: (gsokar) implementation detail missing. Normalize input and weight
        # Ceron: The paper states that it make a difference for large tokens

        # Ceron: capacity of each expert
        # Ceron: we use ceil to allow for per sample token, where the capacity will be 1.
        if self.expert_type == "BIG":
            num_slots = int(
                np.ceil(num_tokens * self.capacity_factor / self.num_experts)
            )
            num_slots_sqrt = np.floor(np.sqrt(num_slots))
            num_slots = int(num_slots_sqrt ** 2)
        else:
            num_slots = int(
                np.ceil(num_tokens * self.capacity_factor / self.num_experts)
            )

        if self.use_random_phi:
            self.phi_weights = torch.normal(size=(token_length, self.num_experts, num_slots), mean=0, std=1)
            # Note that jax.random.normal outputs the standard normal distribution, so we replicate this in PyTorch.
            # Jax link for standard normal: https://jax.readthedocs.io/en/latest/_autosummary/jax.random.normal.html
        else:
            self.phi_weights = nn.Parameter(torch.rand(token_length, self.num_experts, num_slots))
            # TODO: How do you compute fan_in for a three-dimensional tensor?
            # LeCun normal init: stddev = sqrt(1 / fan_in)
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html#jax.nn.initializers.lecun_normal
            # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
            # jax.nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(), dtype=<class 'jax.numpy.float64'>)
            std = np.sqrt(1.0 / self.num_experts)
            torch.nn.init.normal_(self.phi_weights, mean=0, std=std)
            # truncate at two standard deviations
            self.phi_weights.data.clamp_(-2 * std, 2 * std)

        # scale_value is only used if self.normalization=True
        if self.normalization:
            # init as 1 like in the original Ceron code
            self.scale_value = nn.Parameter(torch.ones(1, ))

        # Instantiate the expert networks.
        expert_list = []
        for j in range(self.num_experts):
            expert_model = ExpertModel(
                in_features=token_length,  # PerConv tokenization: Token size is the number of
                # channels
                expert_hidden_size=expert_hidden_size,
                # rng_key=30,
                maintain_token_size=False, noisy=False
            )
            expert_list.append(expert_model)
        # Register expert networks as trainable parameters.
        self.experts = nn.ModuleList(expert_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Creates a lifted version of self.forward_unbatched to perform the forward pass on batched inputs.

        Args:
            x: Tensor of shape [batch_size, num_tokens, token_length].
        Returns:
            y : Tensor of shape [batch_size, num_tokens, token_length] after processing by Soft Mixture
            of Experts layer.
        """
        x = x / 255.0  # Note: Done since forward of backbone is not used
        hidden = self.backbone(x)

        hidden = hidden.permute(0, 2, 3, 1)  # PerConv tokenization
        hidden = torch.flatten(hidden, start_dim=1, end_dim=2)
        hidden = F.relu(hidden)

        if self.fast_forward:
            y = torch.vmap(self.forward_unbatched, in_dims=0, out_dims=0)(hidden)
        else:
            for i in range(hidden.shape[0]):
                y = self.forward_unbatched(hidden[i])
                if i == 0:
                    y_batched = y.unsqueeze(0)
                else:
                    y_batched = torch.cat((y_batched, y.unsqueeze(0)), dim=0)
            y = y_batched
        y = F.relu(y)
        y = torch.nn.Flatten()(y)

        return y

    def forward_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unbatched version of the forward pass for the Soft Mixture of Experts layer. For tokenization types, see p. 2 in
        the paper (https://arxiv.org/pdf/2402.08609). The recommended tokenization type is PerConv, where the number of
        tokens is computed as height times width of the feature map and the token length is equal to the number of
        feature maps.

        Args:
            x: Tensor of shape [num_tokens, token_length].
        Returns:
            y : Tensor of shape [num_tokens, token_length] after processing by Soft Mixture
            of Experts layer.

        Einstein summation convention notation:
        m: number of tokens
        n: number of expert networks employed (self.num_experts)
        p: number of input and output slots (num_slots)
        d: feature dimension (for PerCon tokenization type equal to the number of feature maps in the last convolutional
           layer)
        """

        # Jax:
        # chex.assert_rank(x, 2)
        # chex.assert_rank does not compute the linear algebra rank but instead just uses the dimension of the input
        # tensor:
        # https://github.com/google-deepmind/chex/blob/master/chex/_src/asserts.py
        assert x.dim() == 2

        # Jax:
        # if self.normalization:
        #     # x_normalized = l2_normalize(x, axis=1)
        #     # phi_weights = scale_value[jnp.newaxis, jnp.newaxis, :].repeat(
        #     #     phi_weights.shape[0], axis=0
        #     # ).repeat(phi_weights.shape[1], axis=1).repeat(
        #     #     phi_weights.shape[2], axis=2
        #     # ) * l2_normalize(
        #     #     phi_weights, axis=0
        #     # )
        #     pass
        # else:
        #     x_normalized = x

        if self.normalization:
            # TODO: Why would you repeat self.scale_value like in the original code along the dims instead of using a
            #  scalar, which automatically broadcasts to the entire tensor self.phi_weights?
            x_normalized = l2_normalize(x, dim=1)
            self.phi_weights = self.scale * l2_normalize(self.phi_weights, dim=0)
        else:
            x_normalized = x

        # compute output of gating network
        # size logits: [m, n, p]
        logits = torch.einsum(
            "md,dnp->mnp",
            x_normalized,
            self.phi_weights,
        )

        # Jax code:
        # dispatch_weights = jax.nn.softmax(logits, axis=0)
        # combine_weights = jax.nn.softmax(logits, axis=(1, 2))

        # compute dispatch weights
        dispatch_weights = torch.nn.functional.softmax(logits, dim=0)

        # compute combine weights
        # multi-dim softmax does not seem to exist in PyTorch.
        max_vals = torch.amax(logits, dim=(1, 2))
        exp_logits = torch.exp(logits - max_vals[:, None, None])  # subtract max for numerical stability
        normalizer = exp_logits.sum(dim=(1, 2), keepdim=True)
        combine_weights = exp_logits / normalizer

        # Ceron: Calculate the input tokens for the experts.
        mixture_inputs = torch.einsum("md,mnp->npd", x, dispatch_weights)
        # Ceron: make sure to convert out-of-bounds nans to zeros
        # TODO: Why wouldn't you apply the nan-filter directly to the logits or maybe the combine and dispatch weights?
        #  Moreover, if we use the numerical stable softmax version, where do the nans come from?
        mixture_inputs = torch.nan_to_num(mixture_inputs)

        # TODO: Parallelize this?
        expert_outs = []
        for e, expert in enumerate(self.experts):
            out, _ = expert(mixture_inputs[e, :, :])
            expert_outs.append(out)
        expert_outs = torch.stack(expert_outs, dim=0)

        if self.expert_type == "BIG":
            expert_outs = expert_outs.reshape(self.num_experts, self.num_slots, self.token_length)

        # Ceron: The output tokens are weighted average of all slots.
        outputs = torch.einsum("npd,mnp->md", expert_outs, combine_weights)
        return outputs
