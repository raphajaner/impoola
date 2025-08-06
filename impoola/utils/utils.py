import os
import time
import numpy as np
import wandb
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.nn import Softmax
from torchinfo import summary
from torch_pruning.utils.benchmark import measure_latency
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def get_n_zeros(size, sparsity):
    return int(np.floor(sparsity * size))


def save_agent_to_wandb(config, agent, optimizer, obs_rms, return_rms, metadata={}, aliases=["latest"],
                        full_model=True):
    """Save the state_dict of the agents.py to disk."""
    model_dict = {
        'obs_rms': obs_rms,
        'return_rms': return_rms,
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'config': config
    }
    if full_model:
        model_dict['model'] = agent
    else:
        model_dict['model_state_dict'] = agent.state_dict()

    torch.save(model_dict, f'{wandb.run.dir}/agent.pt')

    model_artifact = wandb.Artifact(
        f"agent-{wandb.run.id}", type="model", description="DRL agent in PyTorch",
        metadata=metadata
    )

    model_artifact.add_file(f'{wandb.run.dir}/agent.pt')
    artifact = wandb.log_artifact(model_artifact, aliases=aliases)  # if is_best else None) "best"
    os.remove(f'{wandb.run.dir}/agent.pt')

    # full_name = wandb.run.entity + '/' + wandb.run.project + '/race_car-' + wandb.run.id
    # for art in wandb.Api().artifact_versions(artifact.type, full_name):
    #     if 'latest' not in art.aliases and 'best_reward' not in art.aliases and 'best_collision' not in art.aliases:
    #         art.delete()
    # runs = api.runs(path="my_entity/my_project")
    # for run in runs:
    #     for artifact in run.logged_artifacts():
    #         if artifact.type == "model":
    #             artifact.delete(delete_aliases=True)


def load_agent_from_wandb(agent, device, agent_id=None, full_model=True):
    if agent_id is None:
        artifact = wandb.use_artifact(f'agent-{wandb.run.id}:latest')
    else:
        artifact = wandb.use_artifact(f'agent-{agent_id}:latest')

    artifact_dir = artifact.download(root=wandb.run.dir)
    print(f"Loaded agent {artifact.source_name} (global step {artifact.metadata['global_step']}).")
    # from  torch._dynamo.eval_frame import OptimizedModule
    # torch.serialization.add_safe_globals([OptimizedModule])
    checkpoint = torch.load(f'{artifact_dir}/agent.pt', map_location=device, weights_only=False)
    os.remove(f'{wandb.run.dir}/agent.pt')
    if full_model:
        agent = checkpoint['model']
    else:
        agent.load_state_dict(checkpoint['model_state_dict'])

    # Set pre_hook again to make sure it is not removed by wrong use of summary that removes hooks
    # from impoola.train.agents import normalize_input
    # agent.network._forward_pre_hooks.clear()
    # agent.network.register_forward_pre_hook(normalize_input)
    # agent.network._forward_pre_hooks.values()

    return agent, checkpoint['obs_rms'], checkpoint['return_rms'], checkpoint['config'], checkpoint[
        'optimizer_state_dict']


@torch.no_grad()
def calculate_global_parameters_number(model, zero_weight_mode=False) -> dict:
    """ Calculate the number of parameters in the model
    Args:
        model: torch.nn.Module
        zero_weight_mode: bool, if True, the params with zero weights are not counted
    Returns:
        n_params_dict: dict, the number of parameters in the model for each layer and the total number of parameters
    """
    n_params_dict = dict()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            if hasattr(module, 'weight'):
                total_zeros = torch.sum(module.weight == 0).item()
                if hasattr(module, 'weight_mask'):
                    total_zeros2 = torch.sum(module.weight_mask == 0).item()
                    assert total_zeros == total_zeros2, "The number of zeros in the weight and weight_mask should be the same"

                total_elements = module.weight.nelement()
                n_params_dict[name] = total_elements - total_zeros if zero_weight_mode else total_elements
                if hasattr(module, 'bias'):
                    total_zeros = torch.sum(module.bias == 0).item()
                    total_elements = module.bias.nelement()
                    n_params_dict[name] += total_elements - total_zeros if zero_weight_mode else total_elements

    n_params_dict['total'] = sum(n_params_dict.values())

    return n_params_dict


def calculate_global_sparsity(pruned_network_params, base_network_params):
    return {k: 1 - (pruned_network_params[k] / base_network_params[k]) for k in pruned_network_params.keys()}


def measure_latency_agent(agent, envs, device, repeat=5000, warmup=5000, batch_size_max=256, batch_size_min=None,
                          dataset=None):
    with torch.inference_mode():
        run_fn = lambda x, y: agent.forward(y)

        # BATCH SIZE Full
        torch.cuda.empty_cache()  # clear the cache
        if dataset is not None:
            assert batch_size_max <= len(dataset), "batch_size must be less than or equal to the dataset size"
            # if no torch tensor, then make one out of if
            if not isinstance(dataset, torch.Tensor):
                dataset = torch.tensor(dataset)
            example_input = dataset[:batch_size_max].to(device)
        else:
            example_input = 128 * np.ones((batch_size_max,) + envs.single_observation_space.shape).astype(
                envs.single_observation_space.dtype)
            example_input = torch.tensor(example_input).to(device)
        latency_max, _ = measure_latency(agent, example_input, run_fn=run_fn, repeat=repeat, warmup=warmup)

        latency_min = None
        if batch_size_min is not None:
            # BATCH SIZE 1
            time.sleep(2)
            torch.cuda.empty_cache()
            if dataset is not None:
                if not isinstance(dataset, torch.Tensor):
                    dataset = torch.tensor(dataset)
                example_input = dataset[:batch_size_min].to(device)
            else:
                example_input = 128 * np.ones((batch_size_min,) + envs.single_observation_space.shape).astype(
                    envs.single_observation_space.dtype)
                example_input = torch.tensor(example_input).to(device)
            latency_min, _ = measure_latency(agent, example_input, run_fn=run_fn, repeat=repeat, warmup=warmup)
    return latency_max, latency_min


def network_summary(network, input_data, device):
    # Note: torchinfo accounts for masking when and params as _orig (subtracts the masked weights correctly)

    statistics = summary(
        network,
        input_data=input_data, device=device,
        depth=10,  # 2,  # 10,
        col_names=("input_size", "output_size", "num_params", "kernel_size", "params_percent", "mult_adds"),
        verbose=1
    )
    total_params = statistics.total_params
    m_macs = np.round(statistics.total_mult_adds / 1e6, 2)
    param_bytes = statistics.total_param_bytes
    return statistics, total_params, m_macs, param_bytes


def covariance_of_gradient_trace(agent, args, batch, criterion, optimizer):
    # Derived from https://github.com/bbartoldson/GeneralizationStabilityTradeoff/blob/main/utils.py

    n_samples = len(batch[0])
    trace = 0.0
    k = 0
    agent.eval()

    def add_square_grad_to(trace, k):
        for module in agent.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                # Weights
                grad_w = module.weight_orig.grad * module.weight_mask if hasattr(module, 'weight_mask') \
                    else module.weight.grad
                if grad_w is not None:
                    trace += (grad_w.detach() ** 2).sum().cpu().item()
                    k_w = module.weight_mask.sum().cpu().item() if hasattr(module, 'weight_mask') \
                        else module.weight.numel()
                    k += k_w
                # Biases
                grad_b = module.bias_orig.grad * module.bias_mask if hasattr(module, 'bias_mask') else module.bias.grad
                if grad_b is not None:
                    trace += (grad_b.detach() ** 2).sum().cpu().item()
                    k_b = module.bias_mask.sum().cpu().item() if hasattr(module, 'bias_mask') else module.bias.numel()
                    k += k_b
        return trace, k

    for i in range(n_samples):
        optimizer.zero_grad()
        b_obs, b_actions, b_logprobs, b_values, b_returns, b_advantages = batch[0][i:i + 1], batch[1][i:i + 1], \
            batch[2][i:i + 1], batch[3][i:i + 1], \
            batch[4][i:i + 1], batch[5][i:i + 1]
        loss = criterion(agent, args, b_obs, b_logprobs, b_actions, b_values, b_returns, b_advantages, batch[5])

        loss[0].backward()
        trace, k = add_square_grad_to(trace, k)

    agent.train()
    return trace / (k * n_samples + 1e-8)


def plot_kernels_cnn(tensor, num_cols=6, include_titles=True, cmap='Blues'):
    if not tensor.ndim == 4:
        raise Exception("The tensor should be 4D")

    num_kernels = tensor.shape[0]
    num_channels = tensor.shape[1]
    height, width = tensor.shape[2], tensor.shape[3]

    # Flatten the input channels for each kernel
    flattened_tensor = tensor.view(num_kernels, num_channels * height, width)

    # Custom colormap: white -> yellow -> red
    colors = ["white", "yellow", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    num_rows = (num_kernels + num_cols - 1) // num_cols
    fig = plt.figure(figsize=(num_cols * 2, num_rows * 2.5))  # Adjusted figure size

    # Normalize over the entire set of flattened kernels
    flattened_tensor = flattened_tensor.abs().numpy()
    tensor_min = 0  # flattened_tensor.min()
    tensor_max = flattened_tensor.max()
    flattened_tensor = (flattened_tensor - tensor_min) / (tensor_max - tensor_min)

    for i in range(num_kernels):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        im = ax1.imshow(flattened_tensor[i], cmap=cmap, aspect='auto', vmin=0, vmax=1)
        # ax1.axis('off')
        # remove ticks but keep the box
        ax1.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        # Add a title to each subplot if requested
        if include_titles:
            ax1.set_title(f"Kernel {i}", fontsize=8)  # Adjust font size as needed

    # Adjust spacing to account for titles
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    # Adding a color bar with a maximum of 1.0
    cbar = fig.colorbar(im, ax=fig.axes, orientation='vertical', fraction=.01)
    cbar.set_label('Activation Scale', rotation=270, labelpad=15)
    return fig


class StopTimer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0.0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            print("Timer is already running!")

    def stop(self):
        if self.running:
            end_time = time.time()
            self.elapsed_time += end_time - self.start_time
            self.running = False
        else:
            print("Timer is not running!")

    def reset(self):
        self.elapsed_time = 0.0
        self.start_time = None
        self.running = False

    def get_elapsed_time(self):
        if self.running:
            current_time = time.time()
            return self.elapsed_time + (current_time - self.start_time)
        return self.elapsed_time

    def __str__(self):
        return f"Elapsed time: {self.get_elapsed_time()} seconds"


def get_group_shape(DG, group):
    # see the group as one unit and get the shape of the group, by assuming this is a linear layer
    # and getting the input and output shape of the group by defining an equivalent linear layer
    output_modules = []
    for dep, idxs in group:
        if hasattr(dep.target.module, 'weight') or \
                hasattr(dep.target.module, 'bias') and DG.is_out_channel_pruning_fn(dep.handler):
            output_modules.append(dep.target.module)

    if len(output_modules) == 0:
        return [0, 0, 0]
    input_shape = output_modules[0].weight.shape[1]
    output_shape = output_modules[-1].weight.shape[0]

    # get overall number of parameters in the group
    n_weights = 0
    n_bias = 0
    group_dict = dict()
    for module in output_modules:
        group_dict[module] = module
        if hasattr(module, 'weight'):
            n_weights += module.weight.numel()
        if hasattr(module, 'bias'):
            n_bias += module.bias.numel()

    # approx. the kernel shape
    kernel = np.array([output_shape, input_shape, n_weights // (input_shape * output_shape)])
    return kernel, n_weights + n_bias, output_modules


def grad_cam(model, input_tensor, target_class, device):
    """Grad-CAM method for visualizing input saliency."""
    grad_target = torch.zeros(32, 96, 8, 8, device=device)
    grad_target[:, :, :, :] = 1
    next_obs = torch.tensor(input_tensor[:32], device=device, dtype=torch.float32)
    next_obs = next_obs.requires_grad_(True)
    q_values = model.encoder.cnn_backbone(next_obs)
    q_values.backward(grad_target)
    receptive_field = next_obs.grad.abs()

    receptive_field = torch.mean(receptive_field, dim=(0, 1), keepdim=True).squeeze(0)
    receptive_field = receptive_field / torch.max(receptive_field)

    receptive_field = receptive_field.permute((1, 2, 0))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    # cmap = LinearSegmentedColormap.from_list('GR', ['k', 'tab:orange', 'red', '#8000FF'])
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    ax.imshow(receptive_field.cpu().numpy(), cmap='binary')  # , cmap=cmap, norm=norm)
    plt.show()

    # plot the orig image
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # plot the orig image
    ax.imshow(next_obs[0].detach().cpu().permute(1, 2, 0).numpy() / 255, cmap='binary')
    plt.show()


@torch.inference_mode()
def calc_translation_sensitivity(model, obs_generator, device, t_pixels=8):
    """Calculate the sensitivity of the model to translations."""

    # Create a translation map that moves the image up to 10 pixels in each direction
    translation_map_actor = torch.zeros(t_pixels * 2 + 1, t_pixels * 2 + 1, device=device)
    translation_map_critic = torch.zeros(t_pixels * 2 + 1, t_pixels * 2 + 1, device=device)
    avg_score_actor = 0
    avg_score_critic = 0

    # Get the output of the model
    original_obs = torch.tensor(obs_generator(0, 0), requires_grad=False).to(device).clone()
    original_logits, orig_value = model(original_obs)

    original_probs = original_logits.log_softmax(dim=1).clone()
    # Softmax  / LogSoftmax
    # original_logit_idx = original_logits.argmax(dim=1).clone()
    # original_prob = original_logits.gather(1, original_logit_idx.unsqueeze(1))


    # pre-allocated translated obs
    translated_obs_all = torch.zeros((t_pixels * 2 + 1, t_pixels * 2 + 1) + original_obs.shape, requires_grad=False)
    for x_shift in range(-t_pixels, t_pixels + 1):
        for y_shift in range(-t_pixels, t_pixels + 1):
            translated_obs_all[t_pixels + y_shift, t_pixels + x_shift] = torch.tensor(obs_generator(y_shift, x_shift))

    for x_shift in range(-t_pixels, t_pixels + 1):
        for y_shift in range(-t_pixels, t_pixels + 1):
            translated_obs = translated_obs_all[t_pixels + y_shift, t_pixels + x_shift].to(device).clone()
            translated_logits, translated_value = model(translated_obs)

            # import Categorial distribution
            translated_probs = Categorical(logits=translated_logits).probs
            # translated_probs = translated_logits.softmax(dim=1)
            score_actor = torch.norm(translated_probs - original_probs, p=1, dim=1, keepdim=True)

            score_actor = score_actor.mean()
            # Do not use anymore
            score_critic = torch.zeros_like(score_actor)  # score_critic.mean()

            translation_map_actor[t_pixels + y_shift, t_pixels + x_shift] = score_actor
            translation_map_critic[t_pixels + y_shift, t_pixels + x_shift] = score_critic
            avg_score_actor += score_actor.cpu().item()
            avg_score_critic += score_critic.cpu().item()

    avg_score_actor /= translation_map_actor.numel()
    avg_score_critic /= translation_map_critic.numel()

    norm_value = 1
    translation_map_actor = translation_map_actor.cpu().numpy() / norm_value
    translation_map_critic = translation_map_critic.cpu().numpy() / norm_value

    return translation_map_actor, translation_map_critic
