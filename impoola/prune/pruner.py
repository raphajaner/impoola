import functools
import typing
import warnings
import torch

import wandb
import numpy as np
from matplotlib import pyplot as plt

from torch_pruning.pruner import GroupNormPruner
from torch import nn
from torch_pruning.dependency import Group
import torch_pruning as tp

import impoola.prune.pytorch_prune as prune
from impoola.utils.schedules import polynomial_scheduler

from torch_pruning.pruner import BasePruningFunc


def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            try:
                prune.remove(module, name="weight")
                prune.remove(module, name="bias")
            except ValueError:
                # print(f'No pruning to remove for {name} and {module}')
                pass
    return model

class PyTorchBasePruner:
    def __init__(self, model, iterative_steps, iterative_pruning_ratio_scheduler, pruning_ratio, ignored_layers=[]):
        self.model = model
        self.iterative_steps = iterative_steps
        self.per_step_pruning_ratio = iterative_pruning_ratio_scheduler(pruning_ratio, iterative_steps)
        self.pruning_ratio = pruning_ratio
        self.iteration = 0
        self.pruning_func_weights = None
        self.pruning_func_bias = None
        self.prune_bias_independently = None
        self.ignored_layers = ignored_layers
        self.init_weight_dict_model = dict()
        self.init_bias_dict_model = dict()
        self.remaining_weight_dict_model = dict()
        self.pruning_ratio_per_layer = True
        self.target_dict_weight = dict()
        self.target_dict_bias = dict()

    def separate_pruning_schedule(self, iterative_steps, pruning_schedule, power_dict, prune_steps):

        if not isinstance(self.pruning_ratio, dict):
            filter_pruning_ratio = dict()
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module,
                                                               nn.Linear):  # and module not in self.ignored_layers:
                    filter_pruning_ratio[module] = self.pruning_ratio

        pruning_ratio_dict = dict()
        for module in filter_pruning_ratio.keys():
            if module.__class__ in power_dict:
                pruning_ratio_dict[module] = polynomial_scheduler(
                    filter_pruning_ratio[module],
                    iterative_steps,
                    start_step=pruning_schedule[0],
                    end_step=pruning_schedule[1],
                    power=power_dict[module.__class__],
                    prune_steps=prune_steps
                )

        self.per_step_pruning_ratio = pruning_ratio_dict
        # self.plot_pruning_ratio_dict()

    def step(self, current_sparsity):
        self.iteration += 1
        did_prune = False

        # if empty dict
        if not self.target_dict_weight:
            for name, n_weight in self.init_weight_dict_model.items():
                # if self.per_step_pruning_ratio is a dicht, we have different pruning ratios for each layer
                if isinstance(self.per_step_pruning_ratio, dict):
                    target_dict_weight_int = (np.array(self.per_step_pruning_ratio[name]) * n_weight).astype(int)
                    target_dict_bias_int = (
                            np.array(self.per_step_pruning_ratio[name]) * self.init_bias_dict_model[name]).astype(int)

                else:
                    target_dict_weight_int = (np.array(self.per_step_pruning_ratio) * n_weight).astype(int)
                    target_dict_bias_int = (
                            np.array(self.per_step_pruning_ratio) * self.init_bias_dict_model[name]).astype(int)

                self.target_dict_weight[name] = np.diff(target_dict_weight_int, prepend=0)
                self.target_dict_bias[name] = np.diff(target_dict_bias_int, prepend=0)

        if self.iteration > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return did_prune

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and module not in self.ignored_layers:

                pruning_amount_weight_int = self.target_dict_weight[module][self.iteration]
                pruning_amount_bias_int = self.target_dict_bias[module][self.iteration]

                if pruning_amount_weight_int <= 0 and pruning_amount_bias_int <= 0:
                    continue

                # WEIGHT, topk are the indices of the weights that are removed
                _, topk = self.pruning_func_weights(module, name="weight", amount=pruning_amount_weight_int,
                                                    importance_scores=module.weight)
                # assert len(topk) == pruning_amount_weight_int, f"Pruning did not work for {name}."

                # BIAS
                if self.prune_bias_independently:
                    self.pruning_func_bias(module, name="bias", amount=pruning_amount_bias_int,
                                           importance_scores=module.bias)
                else:
                    _, topk2 = self.pruning_func_bias(module, name="bias", amount=len(topk), topk=topk)
                    assert set(topk.cpu().numpy()) == set(topk2.cpu().numpy()), f"Pruning did not work for {name}."

                did_prune = True
                self.remaining_weight_dict_model[module] -= len(topk) if topk is not None else 0

                wandb.log({f"n_filters/{name}": self.remaining_weight_dict_model[module]}, commit=False)

        return did_prune

    def remove_pruning(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and module not in self.ignored_layers:
                try:
                    prune.remove(module, name="weight")
                    prune.remove(module, name="bias")
                except ValueError:
                    print(f'No pruning to remove for {name} and {module}')

    def plot_pruning_ratio_dict(self):
        fig_plot = plt.figure()
        averaged_sparsity_distribution = np.zeros(self.iterative_steps + 1)
        for module, sparsity_vector in self.per_step_pruning_ratio.items():
            color = np.random.rand(3)
            plt.plot(sparsity_vector, label=module, color=color)
            averaged_sparsity_distribution += sparsity_vector
        averaged_sparsity_distribution /= len(self.per_step_pruning_ratio)
        plt.plot(averaged_sparsity_distribution, color='red', linewidth=2, label='average')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        wandb.log({"charts/sparsity_distribution": [wandb.Image(fig_plot)]})
        plt.show()


class UnstructuredRandomPruning(PyTorchBasePruner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_func_weights = prune.random_unstructured
        self.pruning_func_bias = prune.random_unstructured
        self.prune_bias_independently = True
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.init_weight_dict_model[module] = module.weight.numel()
                self.init_bias_dict_model[module] = module.bias.numel()
                wandb.log({f"n_filters/{name}": self.init_weight_dict_model[name]}, commit=False)
        self.remaining_weight_dict_model = self.init_weight_dict_model.copy()


class UnstructuredNormPruning(PyTorchBasePruner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_func_weights = prune.l1_unstructured
        self.pruning_func_bias = prune.l1_unstructured
        self.prune_bias_independently = True
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.init_weight_dict_model[module] = module.weight.numel()
                self.init_bias_dict_model[module] = module.bias.numel()
                wandb.log({f"n_filters/{name}": self.init_weight_dict_model[module]}, commit=False)
        self.remaining_weight_dict_model = self.init_weight_dict_model.copy()


class StructuredNormPruning(PyTorchBasePruner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pruning_func_weights = functools.partial(prune.ln_structured, n=1, dim=0)
        self.pruning_func_bias = prune.l1_unstructured
        self.prune_bias_independently = False
        # ignore layers are handled in the base class
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.init_weight_dict_model[module] = module.weight.shape[0]
                self.init_bias_dict_model[module] = module.bias.numel()
                wandb.log({f"n_filters/{name}": self.init_weight_dict_model[module]}, commit=False)
        self.remaining_weight_dict_model = self.init_weight_dict_model.copy()


def _prune_parameter_and_grad_scaled(self, weight, keep_idxs, pruning_dim):
    pruned_weight = torch.nn.Parameter(
        torch.index_select(weight, pruning_dim, torch.LongTensor(keep_idxs).to(weight.device).contiguous()))

    if pruning_dim == 0:
        # rescale the remaining weights to keep the same magnitude as the original weights
        # only do this for outgoing weights, we assume that layers are never transposed (!!!) by checking the dim only
        scaler = torch.clip(weight.norm(p=2) / pruned_weight.norm(p=2), 1)
        # scale = weight.numel() / pruned_weight.numel()
        pruned_weight.data *= scaler

    if weight.grad is not None:
        pruned_weight.grad = torch.index_select(weight.grad, pruning_dim, torch.LongTensor(keep_idxs).to(weight.device))
    return pruned_weight.to(weight.device)


class ResNetGroupNormPruner(GroupNormPruner):
    def __init__(self, rescale_params_after_prune, *args, **kwargs):
        if rescale_params_after_prune:
            BasePruningFunc._prune_parameter_and_grad = _prune_parameter_and_grad_scaled

        super().__init__(*args, **kwargs)

        # get the input conv layers of all resnet blocks
        first_res_layers = []
        for name, layer in self.model.named_modules():
            if name.endswith('.conv') or name.endswith('.conv_mid'):
                first_res_layers.append(layer)
        self.first_res_layers = first_res_layers

    @property
    def iteration(self):
        return self.current_step

    @iteration.setter
    def iteration(self, value):
        self.current_step = value

    def step(self, interactive=False) -> typing.Union[typing.Generator, None]:
        self.current_step += 1
        pruning_method = self.prune_global if self.global_pruning else self.prune_local

        if interactive:  # yield groups for interactive pruning
            out = pruning_method()
            return out
        else:
            for group in pruning_method():
                group.prune(record_history=False)

    def get_all_groups(self):
        groups = self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types)
        for group in groups:
            # if the group contains one of the encoder modules in self.first_res_layers and the handler is to prune
            # its output, then we need to set this module as the root node for the group. This is because the
            # encoder modules are the first layers in the network, and we want to prune them first.
            for dep, idxs in group:
                if dep.target.module in self.first_res_layers and self.DG.is_out_channel_pruning_fn(dep.handler):
                    group = self.DG.get_pruning_group(dep.target.module, dep.handler, idxs)
                    break
            yield group

    def prune_local(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        for group in self.get_all_groups():
            if self._check_pruning_ratio(group):  # check pruning ratio
                ##################################
                # Compute raw importance score
                ##################################

                module = group[0][0].target.module
                pruning_fn = group[0][0].handler
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group)
                if imp is None:
                    continue

                ##################################
                # Compute the number of dims/channels to prune
                ##################################
                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    current_channels = self.DG.get_out_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_out_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                else:
                    current_channels = self.DG.get_in_channels(module)
                    target_pruning_ratio = self.get_target_pruning_ratio(module)
                    n_pruned = current_channels - int(
                        self.layer_init_in_ch[module] *
                        (1 - target_pruning_ratio)
                    )
                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                ##################################
                # collect pruning idxs
                ##################################
                pruning_idxs = []
                _is_attn, qkv_layers = self._is_attn_group(group)
                group_size = current_channels // ch_groups
                # dims/channels
                if n_pruned > 0:
                    if (self.prune_head_dims and _is_attn) or (not _is_attn):
                        n_pruned_per_group = n_pruned // ch_groups
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        if n_pruned_per_group > 0:
                            for chg in range(ch_groups):
                                sub_group_imp = imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size  # offset
                                pruning_idxs.append(sub_pruning_idxs)
                else:  # no channel grouping
                    imp_argsort = torch.argsort(imp)
                    pruning_idxs.append(imp_argsort[:n_pruned])

                if len(pruning_idxs) == 0:
                    continue
                pruning_idxs = torch.unique(torch.cat(pruning_idxs, 0)).tolist()
                group = self.DG.get_pruning_group(module, pruning_fn, pruning_idxs)

                if self.DG.check_pruning_group(group):
                    yield group

    def prune_global(self) -> typing.Generator:
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        for group in self.get_all_groups():
            if self._check_pruning_ratio(group):
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group)  # raw importance score
                group_size = len(imp) // ch_groups
                if imp is None:
                    continue
                if ch_groups > 1:
                    # Corresponding elements of each group will be removed together.
                    # So we average importance across groups here. For example:
                    # imp = [1, 2, 3, 4, 5, 6] with ch_groups=2.
                    # We have two groups [1,2,3] and [4,5,6].
                    # The average importance should be [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
                    dim_imp = imp.view(ch_groups, -1).mean(dim=0)
                else:
                    # no grouping
                    dim_imp = imp
                global_importance.append((group, ch_groups, group_size, dim_imp))

        if len(global_importance) == 0:
            return

        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################

        # Find the threshold for global pruning
        if len(global_importance) > 0:
            concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
            target_pruning_ratio = self.per_step_pruning_ratio[self.current_step]
            n_pruned = len(concat_imp) - int(
                self.initial_total_channels *
                (1 - target_pruning_ratio)
            )
            if n_pruned > 0:
                topk_imp, _ = torch.topk(concat_imp, k=n_pruned, largest=False)
                thres = topk_imp[-1]

        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(
                pruning_fn) else self.DG.get_in_channels

            # Prune feature dims/channels
            pruning_indices = []
            if len(global_importance) > 0 and n_pruned > 0:
                if ch_groups > 1:  # re-compute importance for each channel group if channel grouping is enabled
                    n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
                    if n_pruned_per_group > 0:
                        if self.round_to:
                            n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)
                        _is_attn, _ = self._is_attn_group(group)
                        if not _is_attn or self.prune_head_dims == True:
                            raw_imp = self.estimate_importance(group)  # re-compute importance
                            for chg in range(
                                    ch_groups):  # determine pruning indices for each channel group independently
                                sub_group_imp = raw_imp[chg * group_size: (chg + 1) * group_size]
                                sub_imp_argsort = torch.argsort(sub_group_imp)
                                sub_pruning_idxs = sub_imp_argsort[:n_pruned_per_group] + chg * group_size
                                pruning_indices.append(sub_pruning_idxs)
                else:
                    _pruning_indices = (imp <= thres).nonzero().view(-1)
                    imp_argsort = torch.argsort(imp)
                    if len(_pruning_indices) > 0 and self.round_to:
                        n_pruned = len(_pruning_indices)
                        current_channels = get_channel_fn(module)
                        n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                        _pruning_indices = imp_argsort[:n_pruned]
                    pruning_indices.append(_pruning_indices)

            if len(pruning_indices) == 0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            if self.DG.check_pruning_group(group):
                yield group

    def prune_local2(self) -> typing.Generator:
        # wrong pruning function, do not use!
        if self.current_step > self.iterative_steps:
            warnings.warn("Pruning exceed the maximum iterative steps, no pruning will be performed.")
            return

        total_pruned = 0
        for group in self.get_all_groups():
            module = group[0][0].target.module
            name = group[0][0].target.name.split(' ')[0]

            # get the target pruning ratio for the current step
            target_pruning_ratio = self.pruning_ratio_dict.get(module, [0] * (self.current_step + 1))[self.current_step]
            print(f"Pruning {name} with target ratio {target_pruning_ratio}")
            # target_pruning_ratio = self.pruning_ratio_dict.get(module, self.per_step_pruning_ratio)[self.current_step]
            target_pruning_ratio = min(target_pruning_ratio, self.max_pruning_ratio)

            if target_pruning_ratio > 0 and self._check_pruning_ratio(group):
                pruning_fn = group[0][0].handler

                # compute the number of dims/channels to prune
                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    current_channels = self.DG.get_out_channels(module)
                    init_channels = self.layer_init_out_ch[module]
                else:
                    current_channels = self.DG.get_in_channels(module)
                    init_channels = self.layer_init_in_ch[module]
                n_pruned = current_channels - int(init_channels * (1 - target_pruning_ratio))

                # round to the nearest multiple of round_to
                if self.round_to:
                    n_pruned = self._round_to(n_pruned, current_channels, self.round_to)

                if n_pruned > 0:
                    total_pruned += n_pruned

                    # activation = get_relu_activations_from_hooks(self.model)
                    # activation = get_activations_from_hooks(self.model)

                    group = get_pruning_group_step(self.DG, group, n_pruned, imp_type=self.importance)
                    yield group
        print(f"Pruned {total_pruned} filters in step {self.current_step}")

    def set_local_global_distribution(self, pruning_schedule, filter_pruning_ratio, power=3):
        warmup_steps = int(np.floor(pruning_schedule[0] * self.iterative_steps))

        finetune_steps = int(np.floor((1 - pruning_schedule[1]) * self.iterative_steps))

        pruning_phase_steps = self.iterative_steps - warmup_steps - finetune_steps
        warmup_steps += 1  # pruner counting starts at 1

        sparsity_distribution = dict()

        prunable_filter_dict = dict()
        prunable_weight_per_filter_dict = dict()
        filter_dict = dict()

        all_groups = list(self.get_all_groups())
        total_prunable_filters = 0
        total_filters = 0

        total_prunable_weights = 0
        total_weights = 0

        total_weights_per_filter = 0

        for i, group in enumerate(all_groups):

            root_module = group[0][0].target.module
            root_name = group[0][0].target.name.split(' ')[0]

            if isinstance(filter_pruning_ratio, dict):
                # TODO: check if this is correct
                filter_pruning_ratio = filter_pruning_ratio[root_module]

            n_filter = root_module.weight.shape[0]
            n_prunable_filters = int(np.floor(root_module.weight.shape[0] * filter_pruning_ratio))
            prunable_weights_per_filter = int(np.sum(root_module.weight[0].nelement()))

            prunable_filter_dict[root_name] = n_prunable_filters
            filter_dict[root_module] = n_filter
            prunable_weight_per_filter_dict[root_name] = prunable_weights_per_filter

            total_prunable_filters += n_prunable_filters
            total_prunable_weights += n_prunable_filters * prunable_weights_per_filter

            total_filters += root_module.weight.shape[0]
            total_weights += n_filter * prunable_weights_per_filter

            total_weights_per_filter += prunable_weights_per_filter

            sparsity_distribution[root_name] = np.zeros(pruning_phase_steps)

        sum_filter_pruning_ratio = total_prunable_filters / total_filters
        sum_weights_pruning_ratio = total_prunable_weights / total_weights

        steps_pruning_ratio = polynomial_scheduler(sum_weights_pruning_ratio, pruning_phase_steps, start_step=0,
                                                   end_step=1, power=power, prune_steps=None)

        # steps_pruning_ratio = self.separate_pruning_schedule(
        #     iterative_steps=pruning_phase_steps,
        #     filter_pruning_ratio={root_module: sum_weights_pruning_ratio},
        #     pruning_schedule=[0, 1])[root_module]

        # import pdb
        # pdb.set_trace()

        # set steps_pruning_ratio so that we only take every x value and fill inbetween the values with the value from before, e.g., [1,2,3,4,5,6] -> [1,1,1,4,4,4,5...]
        # actual_steps = 10
        #
        # for i in range(1, len(steps_pruning_ratio)):
        #     if i % ((pruning_phase_steps - 1) // actual_steps) != 0:
        #         steps_pruning_ratio[i] = steps_pruning_ratio[i - 1]
        #     else:
        #         steps_pruning_ratio[i] = steps_pruning_ratio[i]

        # make global schedule out of the steps pruning ratio
        steps_pruning_ratio = np.array(steps_pruning_ratio[1:])
        global_schedule = np.floor(steps_pruning_ratio * total_filters)
        global_schedule_w = np.floor(steps_pruning_ratio * total_weights)

        n_params_per_filter_grou = get_n_params_per_filter_group(self)

        # pruning_schedule = allocate_pruning_to_layers(global_schedule, all_groups, list(prunable_filter_dict.values()))
        pruning_schedule = allocate_pruning_to_layers(global_schedule, all_groups,
                                                      list(prunable_filter_dict.values()),
                                                      list(prunable_weight_per_filter_dict.values()))

        # make a dict of the pruning schedule
        pruning_schedule_dict = dict()
        for idx, root_name in enumerate(filter_dict.keys()):
            pruning_schedule_dict[root_name] = pruning_schedule[:, idx]

        pruning_ratio_dict = dict()

        for name, local_schedule in pruning_schedule_dict.items():
            sparsity_vector = np.cumsum(local_schedule) / filter_dict[name]
            sparsity_vector = np.concatenate(
                (np.zeros(warmup_steps), sparsity_vector, np.ones(finetune_steps) * sparsity_vector[-1]))
            pruning_ratio_dict[name] = sparsity_vector

            # sparsity_vector = np.concatenate((, sparsity_vector))
        #     sparsity_vector = np.concatenate(
        #         (np.zeros(warmup_steps), sparsity_vector, np.ones(finetune_steps) * final_pruning_ratio))
        #
        #     for idx, module in enumerate(all_output_modules):
        #         sparsity_distribution[module] = sparsity_vector
        # TODO: Is wrong as we need to set the dirct with the module and not the name!!!
        self.pruning_ratio_dict = pruning_ratio_dict
        self.plot_pruning_ratio_dict()

    def separate_pruning_schedule(self, iterative_steps, pruning_schedule, filter_pruning_ratio, power_dict):

        if not isinstance(filter_pruning_ratio, dict):
            filter_pruning_ratio = dict()
            for group in self.get_all_groups():
                module = group[0][0].target.module
                filter_pruning_ratio[module] = filter_pruning_ratio

        pruning_ratio_dict = dict()
        for module in filter_pruning_ratio.keys():
            if module.__class__ in power_dict:
                pruning_ratio_dict[module] = polynomial_scheduler(
                    filter_pruning_ratio[module],
                    iterative_steps,
                    start_step=pruning_schedule[0],
                    end_step=pruning_schedule[1],
                    power=power_dict[module.__class__],
                )

        self.pruning_ratio_dict = pruning_ratio_dict

        self.plot_pruning_ratio_dict()

    def plot_pruning_ratio_dict(self):
        fig_plot = plt.figure()
        averaged_sparsity_distribution = np.zeros(self.iterative_steps + 1)
        for module, sparsity_vector in self.pruning_ratio_dict.items():
            color = np.random.rand(3)
            plt.plot(sparsity_vector, label=module, color=color)
            averaged_sparsity_distribution += sparsity_vector
        averaged_sparsity_distribution /= len(self.pruning_ratio_dict)
        plt.plot(averaged_sparsity_distribution, color='red', linewidth=2, label='average')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
        wandb.log({"charts/sparsity_distribution": [wandb.Image(fig_plot)]})
        plt.show()


@torch.inference_mode
def get_pruning_group_step(DG, group, n_pruned, imp_type='l1'):
    dep_pruned = []
    updated_group = Group()

    imp_type = 'activation'

    dep = group[0][0]

    # Recalculate the importance of the channels for output channels, they are never related (only how many are
    # pruned or the input channels)
    if hasattr(dep.target.module, 'weight') and dep.handler in [tp.prune_conv_out_channels,
                                                                tp.prune_linear_out_channels]:

        # Account for different shapes, we take the output channels for importance calculation
        shape = dep.target.module.weight.shape
        if imp_type in ['l1', 'l2', 'activation']:
            imp_object = dep.target.module.weight
        elif imp_type in ['grad', 'taylor']:
            imp_object = dep.target.module.weight.grad
        # elif imp_type == 'activation':
        # imp_type = dep.target.module.weight
        else:
            raise NotImplementedError(f'Not implemented for importance type {imp_type}')
        activation = dep.target.module.activation_

        # define it using the norm function so that we can switch between l1 and l2
        if imp_type in ['l1', 'grad', 'taylor', 'activation']:
            p = 1
        elif imp_type == 'l2':
            p = 2
        else:
            raise NotImplementedError(f'Not implemented for importance type {imp_type}')

        # Calculate the importance of the weights
        if len(shape) == 4:
            imp_w = torch.norm(imp_object.view(shape[0], -1), p=p, dim=1)
        elif len(shape) == 2:
            imp_w = torch.norm(imp_object, p=p, dim=1)
        else:
            raise NotImplementedError(f'Not implemented for shape {shape}')

        root_pruning_idxs_w = torch.topk(imp_w, n_pruned, largest=False).indices.cpu().numpy()

        # Calculate the importance of the activations
        if len(shape) == 4:
            # imp = torch.norm(imp_object, p=p, dim=0).view(shape[0], -1).norm(p, dim=1)
            imp_object_a = activation

            # Take the relu activations
            imp_object_a = torch.clamp(imp_object_a, 0)
            imp_a = imp_object_a.flatten(start_dim=2).norm(p, dim=-1)
            # imp_a = imp_a.mean(dim=0)
            # Calculate the entropy of the activations
            probs = torch.softmax(imp_a, dim=1)
            imp_a = -torch.sum(probs * torch.log(probs), dim=0)

        elif len(shape) == 2:
            # imp = torch.norm(imp_object, p=p, dim=0).view(shape[0], -1).norm(p, dim=1)
            imp_object_a = activation

            # Take the relu activations
            imp_object_a = torch.clamp(imp_object_a, 0)
            # imp_a = imp_object_a.mean(dim=0)
            # Calculate the entropy of the activations
            probs = torch.softmax(imp_object_a, dim=1)
            imp_a = -torch.sum(probs * torch.log(probs), dim=0)
        else:
            raise NotImplementedError(f'Not implemented for shape {shape}')

        root_pruning_idxs_a = torch.topk(imp_a, n_pruned, largest=False).indices.cpu().numpy()
        root_pruning_idxs = root_pruning_idxs_a

        # get indices of the dormant neurons, they have an imp of 0 or smaller
        non_dormant_idxs = torch.nonzero(imp_a > 0).squeeze()
        dormant_idxs = torch.nonzero(imp_a <= 0).squeeze()

        # related_target = []
        # for g in group[i:]:
        #     d, _ = g
        #     if hasattr(d.target.module, 'weight') and d.handler in [tp.prune_conv_in_channels,
        #                                                             tp.prune_linear_in_channels]:
        #         related_target.append(d)
        #
        # # if len(related_target)==2: # and related_target[-1].handler == tp.prune_linear_in_channels:
        # #     import pdb
        # #     pdb.set_trace()
        #
        # related_imp = []
        # for t in related_target:
        #     related_weight = t.target.module.weight
        #     if len(related_weight.shape) == 4:
        #         related_imp.append(related_weight.norm(p, dim=0).view(shape[0], -1).norm(p, dim=1))
        #     elif len(related_weight.shape) == 2:
        #         fraction_idx = int(related_weight.shape[1] / shape[0])
        #         related_imp.append(
        #             related_weight.view(related_weight.shape[0], -1, shape[0]).norm(p, dim=0).norm(p, dim=0))
        #
        # related_imp_sum = 0
        # for i in related_imp:
        #     related_imp_sum += i / i.sum()

        # Normalize the importance
        # imp /= imp.sum()

        # print(imp_object.shape)

        # imp += related_imp[0] / related_imp[0].sum()

        # get the idices sorted according to the importance

        # if we have dormant neurons, the select n_pruned from the dormant neurons according to the importance.
        # this means the one we select have the smallest importance among the dormant neuron
        # if len(imp_a) > len(non_dormant_idxs):
        #     root_pruning_idxs_all = torch.argsort(imp, descending=False).cpu().numpy()
        #     root_pruning_idxs = []
        #     # Get the indices of the smallest importance
        #     for idx in root_pruning_idxs_all:
        #         if idx not in non_dormant_idxs:
        #             root_pruning_idxs.append(idx)
        #         if len(root_pruning_idxs) == n_pruned:
        #             break
        #
        #     if len(root_pruning_idxs) < n_pruned:
        #         print(f"not enough non dormants: {len(root_pruning_idxs)} < {n_pruned}")
        #         n_remaining = n_pruned - len(root_pruning_idxs)
        #         # Get the indices of the smallest importance
        #         for idx in root_pruning_idxs_all:
        #             if idx not in root_pruning_idxs:
        #                 root_pruning_idxs.append(idx)
        #             if len(root_pruning_idxs) == n_pruned:
        #                 break
        #
        # else:
        #     # root_pruning_idxs = root_pruning_idxs_all[:n_pruned]
        # root_pruning_idxs = torch.topk(imp_a, n_pruned, largest=False).indices.cpu().numpy()

        # assert len(root_pruning_idxs) == n_pruned, 'something is wrong with the pruning idxs'
        # check that the idx are uniqure
        if not len(set(root_pruning_idxs)) == n_pruned or not len(root_pruning_idxs) == n_pruned:
            import pdb
            pdb.set_trace()

        # Get the indices of the smallest importance
        # root_pruning_idxs = torch.topk(imp, n_pruned, largest=False).indices.cpu().numpy()

        # Get the indices of the smallest importances and make sure that this is a

        # We reset the group to now have the current module as root
        group = DG.get_pruning_group(dep.target.module, dep.handler, root_pruning_idxs)
        i, d = 0, len(group)
        dep, idxs = group[i]
        # tp convention that the target_name is the name of the module split by ' '
        target_name = dep.target.name.split(' ')[0]
        dep_pruned.append([target_name, dep.handler.__name__])

        updated_group.add_dep(dep=dep, idxs=idxs)
        i += 1

    return updated_group


@torch.inference_mode
def get_zero_weight_group_step(DG, group):
    dep_pruned = []
    updated_group = Group()

    imp_type = 'activation'

    # Prune as maximum over all groups
    i, d = 0, len(group)

    while i < d:
        dep, idxs = group[i]

        # if we already pruned this dependency, we start with the next group
        do_break = False
        for dep_p in dep_pruned:
            if dep_p[0] == dep.target.name.split(' ')[0] and dep_p[1] == dep.handler.__name__:
                do_break = True
                break
        if do_break:
            break  # break the while loop

        # Recalculate the importance of the channels for output channels, they are never related (only how many are
        # pruned or the input channels)
        if hasattr(dep.target.module, 'weight') and dep.handler in [tp.prune_conv_out_channels,
                                                                    tp.prune_linear_out_channels]:

            # Account for different shapes, we take the output channels for importance calculation
            shape = dep.target.module.weight.shape
            imp_object = dep.target.module.weight

            # define it using the norm function so that we can switch between l1 and l2
            if imp_type in ['l1', 'grad', 'taylor', 'activation']:
                p = 1
            elif imp_type == 'l2':
                p = 2
            else:
                raise NotImplementedError(f'Not implemented for importance type {imp_type}')

            # Calculate the importance of the weights
            if len(shape) == 4:
                imp_w = torch.norm(imp_object.view(shape[0], -1), p=p, dim=1)
            elif len(shape) == 2:
                imp_w = torch.norm(imp_object, p=p, dim=1)
            else:
                raise NotImplementedError(f'Not implemented for shape {shape}')

            # get idxs of neurons where all weights are zero
            root_pruning_idxs_w = torch.nonzero(imp_w == 0).squeeze().cpu().numpy()

            group = DG.get_pruning_group(dep.target.module, dep.handler, root_pruning_idxs_w)
            i, d = 0, len(group)
            dep, idxs = group[i]
            # tp convention that the target_name is the name of the module split by ' '
            target_name = dep.target.name.split(' ')[0]
            dep_pruned.append([target_name, dep.handler.__name__])

        updated_group.add_dep(dep=dep, idxs=idxs)
        i += 1

    return updated_group


def get_n_params_per_filter_group(self):
    n_params_per_filter_group = dict()
    for group in self.get_all_groups():

        # get the layer pruning rate given the filter pruning rate

        # I want to know how many params are removed when I remove one filter of the root module
        pruning_group = self.DG.get_pruning_group(group[0][0].target.module, group[0][0].handler, [0])
        # calculate the number of parameters removed when removing one filter
        n_params_per_pruned_filter = 0
        n_params_per_full_filter = 0
        for dep, idx in pruning_group:
            module = dep.target.module
            pruning_fn = dep.handler
            if hasattr(module, 'weight'):
                if self.DG.is_out_channel_pruning_fn(pruning_fn):
                    n_params_per_pruned_filter += module.weight[idx].nelement()
                    n_params_per_full_filter += module.weight.nelement()
                elif self.DG.is_in_channel_pruning_fn(pruning_fn):
                    n_params_per_pruned_filter += module.weight[:, idx].nelement()
                    n_params_per_full_filter += module.weight.nelement()
            if hasattr(module, 'bias') and module.bias is not None and self.DG.is_out_channel_pruning_fn(
                    pruning_fn):
                n_params_per_pruned_filter += module.bias.nelement()
                n_params_per_full_filter += module.bias.nelement()

        n_params_per_filter_group[group] = (n_params_per_pruned_filter, n_params_per_full_filter)

    # for group, n_params_per_filter in n_params_per_filter_group.items():
    #     print(f"Group: {group[0][0].target} removed params per filter: {n_params_per_filter}")
    return n_params_per_filter_group


def create_global_pruning_schedule(total_params, total_steps):
    # Calculate the ideal number of parameters to prune per step
    params_per_step = total_params / (total_steps + 1e-3)

    # Create the global pruning schedule
    global_schedule = np.full(total_steps, params_per_step)

    # Now discretize the schedule
    global_schedule = np.round(np.cumsum(global_schedule))
    # make sure that the last one is the total_params
    assert global_schedule[-1] == total_params, f"Last element of global schedule is {global_schedule[-1]}"

    return global_schedule


def allocate_pruning_to_layers(global_schedule, layers, prunable_params, weights_per_filter):
    total_steps = len(global_schedule)
    total_params = max(global_schedule)

    weights_per_filter = np.array(weights_per_filter)

    # Initialize the pruning schedule for each layer
    pruning_schedule = np.zeros((total_steps, len(layers)), dtype=int)
    initial_per_layer = np.array(prunable_params)
    remaining_pruned_per_layer = np.array(prunable_params)

    # proportions = [n_l / total_params for n_l in prunable_params]
    # proportions = np.array([1.0 for _ in prunable_params])
    proportions = np.array([(n_l / i_l) for n_l, i_l in zip(remaining_pruned_per_layer, initial_per_layer)])
    proportions /= np.sum(proportions)

    current_cumsum = 0

    for step in range(total_steps):

        current_target = global_schedule[step]
        step_quota = current_target - current_cumsum

        filter_in_step = 0

        if step_quota > 0:
            while step_quota > 0:
                proportions = np.array([(n_l / i_l) for n_l, i_l in zip(remaining_pruned_per_layer, initial_per_layer)])

                sum_proportions = np.sum(proportions)
                if sum_proportions == 0:
                    break

                proportions /= sum_proportions

                # layer_idx = np.random.choice(len(layers), p=proportions)

                layer_idx = np.argmax(proportions)

                pruning_schedule[step, layer_idx] += 1
                remaining_pruned_per_layer[layer_idx] -= 1

                step_quota -= 1
                current_cumsum += 1
                filter_in_step += 1

    # assert np.sum(remaining_pruned_per_layer) == 0, f"Remaining pruned per layer is {remaining_pruned_per_layer}"

    pruning_steps = np.sum(pruning_schedule, axis=1)
    # plot this
    plt.plot(pruning_steps)
    return pruning_schedule


class ZeroWeightPruner(ResNetGroupNormPruner):
    def prune_local(self) -> typing.Generator:
        for group in self.get_all_groups():
            group = get_zero_weight_group_step(self.DG, group)
            yield group
