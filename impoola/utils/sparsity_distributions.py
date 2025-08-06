import numpy as np
from cleanrl.utils.utils import get_group_shape, get_n_zeros


def get_sparsities_erdos_renyi(
        var_shape_dict,
        default_sparsity,
        custom_sparsity_map=None,
        include_kernel=True,
        erk_power_scale=1.0,
):
    """Given the method, returns the sparsity of individual layers as a dict.

    It ensures that the non-custom layers have a total parameter count as the one
    with uniform sparsities. In other words for the layers which are not in the
    custom_sparsity_map the following equation should be satisfied.

    N_i refers to the parameter count at layer i.
    (p_i * eps) gives the sparsity of layer i.

    # eps * (p_1 * N_1 + p_2 * N_2) = (1 - default_sparsity) * (N_1 + N_2)
    # where p_i is np.sum(var_i.shape) / np.prod(var_i.shape)
    # for each i, eps*p_i needs to be in [0, 1].
    Args:
      var_shape_dict: dict, of shape of all Variables to prune.
      default_sparsity: float, between 0 and 1.
      custom_sparsity_map: dict or None, <str, float> key/value pairs where the
        mask correspond whose name is '{key}/mask:0' is set to the corresponding
        sparsity value.
      include_kernel: bool, if True kernel dimension are included in the scaling.
      erk_power_scale: float, if given used to take power of the ratio. Use
        scale<1 to make the erdos_renyi softer.

    Returns:
      sparsities, dict of where keys() are equal to all_masks and sparsities
        masks are mapped to their sparsities.
    """
    if not var_shape_dict:
        raise ValueError('Variable shape dictionary should not be empty')
    if default_sparsity is None or default_sparsity < 0 or default_sparsity > 1:
        raise ValueError('Default sparsity should be a value between 0 and 1.')

    # We have to enforce custom sparsities and then find the correct scaling
    # factor.
    if custom_sparsity_map is None:
        custom_sparsity_map = {}
    is_eps_valid = False

    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    dense_layers = set()
    while not is_eps_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # four layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for var_name, var_shape in var_shape_dict.items():
            n_param = np.prod(var_shape)
            n_zeros = get_n_zeros(n_param, default_sparsity)
            if var_name in dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            elif var_name in custom_sparsity_map:
                # We ignore custom_sparsities in erdos-renyi calculations.
                pass
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                n_ones = n_param - n_zeros
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                if include_kernel:
                    raw_probabilities[var_name] = (np.sum(var_shape) / np.prod(var_shape)) ** erk_power_scale
                else:
                    # n_in, n_out = var_shape[-2:] # for tensorflow
                    n_out, n_in = var_shape[:2]  # for pytorch
                    raw_probabilities[var_name] = (n_in + n_out) / (n_in * n_out)

                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[var_name] * n_param

        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        eps = rhs / divisor
        # If eps * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * eps
        if max_prob_one > 1:
            is_eps_valid = False
            for var_name, raw_prob in raw_probabilities.items():
                if raw_prob == max_prob:
                    dense_layers.add(var_name)
        else:
            is_eps_valid = True

    sparsities = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for var_name, var_shape in var_shape_dict.items():
        n_param = np.prod(var_shape)
        if var_name in custom_sparsity_map:
            sparsities[var_name] = custom_sparsity_map[var_name]
        elif var_name in dense_layers:
            sparsities[var_name] = 0.0
        else:
            probability_one = eps * raw_probabilities[var_name]
            sparsities[var_name] = 1.0 - probability_one
    return sparsities


def get_sparsities_proportional(agent, example_input, pruning_ratio, ignored_layers):
    from torch_pruning import DependencyGraph, ops
    DG = DependencyGraph().build_dependency(
        model=agent,
        example_inputs=example_input,
        forward_fn=lambda x, y: agent.forward(y),
    )
    modules_dict = {}

    # Populate modules_dict with parameters grouped by their corresponding modules
    for i, group in enumerate(DG.get_all_groups(
            ignored_layers=ignored_layers, root_module_types=(ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM)
    )):
        _, group_params, output_modules = get_group_shape(DG, group)
        for module in output_modules:
            if module not in ignored_layers:
                if module not in modules_dict:
                    modules_dict[module] = 0
                modules_dict[module] += group_params // len(output_modules)

    total_params = sum(modules_dict.values())
    mean_params = total_params / len(modules_dict)

    def recursive_distribute(modules_dict, pruning_ratio_dict, remaining_pruning_params, total_params,
                             mean_params, step=0, max_steps=100):
        if step >= max_steps or remaining_pruning_params <= 0:
            return pruning_ratio_dict, remaining_pruning_params

        for module, params in modules_dict.items():
            # Calculate the sparsity for this module
            sparsity = (params / mean_params) ** 0.5 * remaining_pruning_params / total_params
            # Update the pruning ratio for the module, ensuring it stays within [0, 0.95]
            new_sparsity = np.clip(pruning_ratio_dict[module] + sparsity, 0, 0.95)
            pruning_increment = new_sparsity - pruning_ratio_dict[module]
            pruning_ratio_dict[module] = new_sparsity
            remaining_pruning_params -= int(pruning_increment * params)

        if remaining_pruning_params > 0 and step + 1 < max_steps:
            return recursive_distribute(modules_dict, pruning_ratio_dict, remaining_pruning_params,
                                        total_params, mean_params, step=step + 1, max_steps=max_steps)

        return pruning_ratio_dict, remaining_pruning_params

    # Initialize pruning ratios for each module to zero
    pruning_ratio_dict = {k: 0 for k in modules_dict.keys()}
    # Calculate initial remaining pruning parameters
    initial_remaining_pruning_params = int(pruning_ratio * total_params)

    # Run the recursive distribution of pruning ratios
    pruning_ratio_dict, remaining_pruning_params = recursive_distribute(
        modules_dict, pruning_ratio_dict, initial_remaining_pruning_params, total_params, mean_params
    )

    # Optional: Calculate the expected sparsity across all modules
    expected_sparsity = sum(
        [v * p for v, p in zip(pruning_ratio_dict.values(), modules_dict.values())]) / total_params
    print(f"Expected sparsity after pruning: {expected_sparsity:.4f}")
    return pruning_ratio_dict
