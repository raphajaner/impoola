import wandb
from impoola.utils.utils import calculate_global_parameters_number, calculate_global_sparsity


def pruning_step(args, agent, optimizer, pruning_func, pruner, step, zero_weight_mode, base_network_params,
                 current_network_params, global_sparsity, obs=None):

    if args.pruning_type == "ReDo":
        if step % args.redo_interval == 0 and step > 0:
            redo_dict = pruning_func(obs, agent, optimizer, args.redo_tau, re_initialize=True, use_lecun_init=False)
            wandb.log({
                "charts/zero_fraction": redo_dict['zero_fraction'],
                "charts/dormant_fraction": redo_dict['dormant_fraction']
            })
        return False, current_network_params, global_sparsity

    elif args.pruning_type in ["UnstructuredRandom", "UnstructuredNorm"]:

        # if not (pruner.per_step_pruning_ratio[pruner.iteration + 1] > pruner.per_step_pruning_ratio[pruner.iteration]):

        if pruner.iteration == 0:
            pruner.iteration += 1
            return False, current_network_params, global_sparsity

        # Do the pruning step
        did_prune = pruning_func(pruner, optimizer, args, step, global_sparsity)

        pruned_network_params = calculate_global_parameters_number(agent, zero_weight_mode=zero_weight_mode)
        global_sparsity = calculate_global_sparsity(pruned_network_params, base_network_params)
        relative_params_removed = \
            (current_network_params['total'] - pruned_network_params['total']) / current_network_params['total']

        wandb.log({
            "charts/global_sparsity": global_sparsity['total'],
            "charts/total_network_params": pruned_network_params['total'],
            "charts/relative_params_removed": relative_params_removed,
        }, commit=False)

        return did_prune, pruned_network_params, global_sparsity
    else:
        raise ValueError(f"Unsupported pruning type: {args.pruning_type}")

