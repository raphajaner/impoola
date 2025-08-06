# Taken from https://github.com/timoklein/redo/blob/main/src/redo.py

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# redo_tau: float = 0.025  # 0.025 for default, else 0.1
# redo_check_interval: int = 1000
# redo_bs: int = 64
# if global_step % cfg.redo_check_interval == 0:
#                 redo_samples = rb.sample(cfg.redo_bs)
#                 redo_out = run_redo(
#                     redo_samples,
#                     model=q_network,
#                     optimizer=optimizer,
#                     tau=cfg.redo_tau,
#                     re_initialize=cfg.enable_redo,
#                     use_lecun_init=cfg.use_lecun_init,
#                 )
#
#                 q_network = redo_out["model"]
#                 optimizer = redo_out["optimizer"]
#
#                 logs |= {
#                     f"regularization/dormant_t={cfg.redo_tau}_fraction": redo_out["dormant_fraction"],
#                     f"regularization/dormant_t={cfg.redo_tau}_count": redo_out["dormant_count"],
#                     "regularization/dormant_t=0.0_fraction": redo_out["zero_fraction"],
#                     "regularization/dormant_t=0.0_count": redo_out["zero_count"],
#                 }

@torch.no_grad()
def _kaiming_uniform_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Kaiming uniform scheme."""

    # This is adapted from https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
    fan_in = nn.init._calculate_correct_fan(tensor=layer.weight, mode="fan_in")
    gain = nn.init.calculate_gain(nonlinearity="relu", param=math.sqrt(5))
    std = gain / math.sqrt(fan_in)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    layer.weight.data[mask, ...] = torch.empty_like(layer.weight.data[mask, ...]).uniform_(-bound, bound)

    if layer.bias is not None:
        # The original code resets the bias to 0.0 because it uses a different initialization scheme
        # layer.bias.data[mask] = 0.0
        if isinstance(layer, nn.Conv2d):
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)
        else:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            layer.bias.data[mask, ...] = torch.empty_like(layer.bias.data[mask, ...]).uniform_(-bound, bound)


@torch.no_grad()
def _lecun_normal_reinit(layer: nn.Linear | nn.Conv2d, mask: torch.Tensor) -> None:
    """Partially re-initializes the bias of a layer according to the Lecun normal scheme."""

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)

    # This implementation follows the jax one
    # https://github.com/google/jax/blob/366a16f8ba59fe1ab59acede7efd160174134e01/jax/_src/nn/initializers.py#L260
    variance = 1.0 / fan_in
    stddev = math.sqrt(variance) / 0.87962566103423978
    layer.weight[mask] = nn.init._no_grad_trunc_normal_(layer.weight[mask], mean=0.0, std=1.0, a=-2.0, b=2.0)
    layer.weight[mask] *= stddev
    if layer.bias is not None:
        layer.bias.data[mask] = 0.0


@torch.inference_mode()
def _get_activation(name: str, activations: dict[str, torch.Tensor]):
    """Fetches and stores the activations of a network layer."""

    def hook(layer: nn.Linear | nn.Conv2d, input: tuple[torch.Tensor], output: torch.Tensor) -> None:
        """
        Get the activations of a layer with relu nonlinearity.
        ReLU has to be called explicitly here because the hook is attached to the conv/linear layer.
        """
        activations[name] = F.relu(output)
        if 'weight_mask' in [buffer[0] for buffer in layer.named_buffers()]:
            # only select the active neurons
            # reduce all dims except the output dim which is 0
            # check if conv layer
            if isinstance(layer, nn.Conv2d):
                mask_output_nonzero = layer.weight_mask.sum(dim=(1, 2, 3)) != 0
            elif isinstance(layer, nn.Linear):
                mask_output_nonzero = layer.weight_mask.sum(dim=1) != 0
            else:
                raise ValueError("Only Conv2d and Linear layers are supported")
            activations[name] = activations[name][:, mask_output_nonzero]
        else:
            activations[name] = activations[name]

    return hook


@torch.inference_mode()
def _get_redo_masks(activations: dict[str, torch.Tensor], tau: float) -> torch.Tensor:
    """
    Computes the ReDo mask for a given set of activations.
    The returned mask has True where neurons are dormant and False where they are active.
    """
    masks = []
    names = []
    # Remove the layers that are considered output layers like the critic, actor, etc.
    # valid_activations = [(name, activation) for name, activation in list(activations.items()) if
    #                      name not in ["critic", "actor", "q", "q1", "q2", "value", "policy"]]
    valid_activations = [
        (name, activation) for name, activation in list(activations.items()) if
        all(out_name not in name for out_name in ["critic", "actor", "q", "q1", "q2", "value", "policy"])
    ]

    # print all names
    # print([name for name, activation in valid_activations])

    for name, activation in valid_activations:
        # Taking the mean here conforms to the expectation under D in the main paper's formula
        if activation.ndim == 4:
            # Conv layer
            score = activation.abs().mean(dim=(0, 2, 3))
        else:
            # Linear layer
            score = activation.abs().mean(dim=0)

        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        layer_mask = torch.zeros_like(normalized_score, dtype=torch.bool)
        if tau > 0.0:
            layer_mask[normalized_score <= tau] = 1
        else:
            layer_mask[torch.isclose(normalized_score, torch.zeros_like(normalized_score))] = 1
        masks.append(layer_mask)
        names.append(name)
    return masks, names


@torch.no_grad()
def _reset_dormant_neurons(model, redo_masks: torch.Tensor, use_lecun_init: bool):
    """Re-initializes the dormant neurons of a model."""

    # layers = [(name, layer) for name, layer in list(model.named_modules())[1:]]
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) \
                and name not in ["critic", "actor", "q", "q1", "q2", "value", "policy"]:
            layers.append((name, module))

    # assert len(redo_masks) == len(layers) - 1, "Number of masks must match the number of layers"
    assert len(redo_masks) == len(layers), "Number of masks must match the number of layers"
    # print([mask.shape for mask in redo_masks])
    # print([layer[0] for layer in layers])

    # Reset the ingoing weights
    # Here the mask size always matches the layer weight size
    for i in range(len(layers[:-1])):
        # for i in range(len(layers)):
        mask = redo_masks[i]
        layer = layers[i][1]
        next_layer = layers[i + 1][1]
        # Can be used to not reset outgoing weights in the Q-function
        next_layer_name = layers[i + 1][0]

        # Skip if there are no dead neurons
        if torch.all(~mask):
            # No dormant neurons in this layer
            continue
        else:
            print(f"Layer {layers[i][0]} has {mask.sum()} dormant neurons")

        # The initialization scheme is the same for conv2d and linear
        # 1. Reset the ingoing weights using the initialization distribution
        if use_lecun_init:
            _lecun_normal_reinit(layer, mask)
        else:
            _kaiming_uniform_reinit(layer, mask)

        # 2. Reset the outgoing weights to 0
        # NOTE: Don't reset the bias for the following layer or else you will create new dormant neurons
        # To not reset in the last layer: and not next_layer_name == 'q'
        #  name not in ["critic", "actor", "q", "q1", "q2", "value", "policy"]:
        if isinstance(layer, nn.Conv2d) and isinstance(next_layer, nn.Linear):
            # Special case: Transition from conv to linear layer
            # Reset the outgoing weights to 0 with a mask created from the conv filters
            num_repeatition = next_layer.weight.data.shape[1] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repeatition)
            next_layer.weight.data[:, linear_mask] = 0.0
        else:
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            next_layer.weight.data[:, mask, ...] = 0.0

    return model


@torch.no_grad()
def _reset_adam_moments(optimizer: optim.Adam, reset_masks: dict[str, torch.Tensor]) -> optim.Adam:
    """Resets the moments of the Adam optimizer for the dormant neurons."""

    assert isinstance(optimizer, optim.Adam), "Moment resetting currently only supported for Adam optimizer"
    for i, mask in enumerate(reset_masks):
        # Reset the moments for the weights
        optimizer.state_dict()["state"][i * 2]["exp_avg"][mask, ...] = 0.0
        optimizer.state_dict()["state"][i * 2]["exp_avg_sq"][mask, ...] = 0.0
        # NOTE: Step count resets are key to the algorithm's performance
        # It's possible to just reset the step for moment that's being reset
        optimizer.state_dict()["state"][i * 2]["step"].zero_()

        # Reset the moments for the bias
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg"][mask] = 0.0
        optimizer.state_dict()["state"][i * 2 + 1]["exp_avg_sq"][mask] = 0.0
        optimizer.state_dict()["state"][i * 2 + 1]["step"].zero_()

        # Reset the moments for the output weights
        if (
                len(optimizer.state_dict()["state"][i * 2]["exp_avg"].shape) == 4
                and len(optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"].shape) == 2
        ):
            # Catch transition from conv to linear layer through moment shapes
            num_repeatition = optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"].shape[1] // mask.shape[0]
            linear_mask = torch.repeat_interleave(mask, num_repeatition)
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"][:, linear_mask] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg_sq"][:, linear_mask] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["step"].zero_()
        else:
            # Standard case: layer and next_layer are both conv or both linear
            # Reset the outgoing weights to 0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg"][:, mask, ...] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["exp_avg_sq"][:, mask, ...] = 0.0
            optimizer.state_dict()["state"][i * 2 + 2]["step"].zero_()

    return optimizer


@torch.no_grad()
def run_redo(
        obs,
        model,
        optimizer: optim.Adam,
        tau: float,
        re_initialize: bool,
        use_lecun_init: bool,
) -> dict:  # tuple[nn.Module, optim.Adam, float, int]:
    """
    Checks the number of dormant neurons for a given model.
    If re_initialize is True, then the dormant neurons are re-initialized according to the scheme in
    https://arxiv.org/abs/2302.12902

    Returns the number of dormant neurons.
    """
    obs = obs[:256] if len(obs) > 256 else obs

    with torch.inference_mode():
        activations = {}
        activation_getter = partial(_get_activation, activations=activations)

        # Register hooks for all Conv2d and Linear layers to calculate activations
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                handles.append(module.register_forward_hook(activation_getter(name)))

            # Calculate activations
            if hasattr(model.forward, "_torchdynamo_orig_callable"):
                _ = model._torchdynamo_orig_callable(obs)
            else:
                _ = model(obs)

        # Masks for tau=0 logging
        zero_masks, _ = _get_redo_masks(activations, 0.0)
        total_neurons = sum([torch.numel(mask) for mask in zero_masks])
        zero_count = sum([torch.sum(mask) for mask in zero_masks])
        zero_fraction = (zero_count / total_neurons) * 100

        # Calculate the masks actually used for resetting
        masks, names = _get_redo_masks(activations, tau)

        dormant_neurons_per_layer = {name: mask.sum().item() / mask.numel() * 100 for name, mask in zip(names, masks)}
        # if the name contains _orig_mod, remove it form the name
        for name in list(dormant_neurons_per_layer.keys()):
            if '_orig_mod' in name:
                new_name = name.replace('_orig_mod.', '')
                dormant_neurons_per_layer[new_name] = dormant_neurons_per_layer.pop(name)

        dormant_count = sum([torch.sum(mask) for mask in masks])
        dormant_fraction = (dormant_count / sum([torch.numel(mask) for mask in masks])) * 100

        # Re-initialize the dormant neurons and reset the Adam moments
        if re_initialize:
            model = _reset_dormant_neurons(model, masks, use_lecun_init)
            optimizer = _reset_adam_moments(optimizer, masks)

        # Remove the hooks again
        for handle in handles:
            handle.remove()

        return {
            "model": model,
            "optimizer": optimizer,
            "zero_fraction": zero_fraction,
            "zero_count": zero_count,
            "dormant_fraction": dormant_fraction,
            "dormant_count": dormant_count,
            "dormant_neurons_per_layer": dormant_neurons_per_layer,
        }
