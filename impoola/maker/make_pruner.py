from functools import partial
from torch import nn

from impoola.prune.redo import run_redo
from impoola.utils.schedules import polynomial_scheduler
from impoola.prune.pruner import UnstructuredNormPruning


def make_pruner(args, agent, iterative_pruning_steps):
    """ Make the pruner for the agent """
    if args.pruning_type == "ReDo":
        return None, run_redo, False
    elif args.pruning_type == "UnstructuredNorm":

        ignored_layers = [getattr(agent, layer) for layer in ['actor', 'critic', 'value'] if hasattr(agent, layer)]

        # if args.ignore_last_linear:
        #     for layer in reversed(agent.network):
        #         if isinstance(layer, nn.Linear):
        #             ignored_layers.append(layer)
        #             break

        pruning_ratio_scheduler = partial(polynomial_scheduler, start_step=0.2, end_step=0.8, power=3)

        pruner = UnstructuredNormPruning(
            agent,
            iterative_steps=iterative_pruning_steps,
            iterative_pruning_ratio_scheduler=pruning_ratio_scheduler,
            pruning_ratio=0.9,
            ignored_layers=ignored_layers,
        )

        def unstructured_pruning(pruner, optimizer, args, iteration, current_sparsity):
            return pruner.step(current_sparsity)

        return pruner, unstructured_pruning, True
    else:
        raise ValueError(f"Unsupported pruning type: {args.pruning_type}")
