# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy

import wandb
import tyro
import gym
import numpy as np
import torch
import torch.optim as optim

from impoola.utils.utils import network_summary, save_agent_to_wandb, measure_latency_agent
from impoola.maker.make_env import make_an_env
from impoola.train.agents import DQNAgent
from impoola.eval import evaluation
from impoola.prune.redo import run_redo
from impoola.train.train_dqn_agent import train_dqn_agent
from impoola.eval.normalized_score_lists import progcen_easy_hns, progcen_hard_hns, progcen_hns


@dataclass
class Args:
    # General Settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "impoola"
    """the wandb's project name"""
    wandb_entity: str = 'wandb_entity'
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_track_setting: str = "generalization"
    """the track setting of the environment"""
    n_episodes_rollout: int = int(2.5e3)
    """the number of episodes to rollout for evaluation"""
    training_eval_ratio: float = 0.1
    """the ratio of training evaluation"""
    deterministic_rollout: bool = True
    """if toggled, the rollout will be deterministic"""
    measure_latency: bool = False
    """if toggled, the latency of the agent will be measured"""
    compile_agent: bool = False
    """if toggled, the agent will be compiled"""
    normalize_reward: bool = False
    """if toggled, the reward will be normalized"""

    # Algorithm specific arguments
    env_id: str = "bigfish"
    """the id of the environment"""
    distribution_mode: str = "easy"
    """the distribution mode of the environment"""
    total_timesteps: int = int(25e6)
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: Optional[float] = None
    """the target network update rate"""
    target_network_frequency: int = 128 * 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 512
    """the batch size of sample from the reply memory"""
    start_e: float = 1.0
    """the starting epsilon for exploration"""
    end_e: float = 0.025
    """the ending epsilon for explortion"""
    exploration_fraction: float = 0.1
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 250000
    """timestep to start learning"""
    train_frequency: float = 1
    """the frequency of training"""

    # Additional arguments
    double_dqn: bool = True
    """if toggled, double dqn will be enabled"""
    multi_step: int = 3
    """the number of steps to rollout for multi-step learning"""
    prioritized_replay: bool = True
    """if toggled, prioritized replay will be enabled"""
    softmax_exploration: bool = False
    """if toggled, softmax exploration will be enabled"""
    max_grad_norm: float = 10.0  # 0.5
    """the maximum gradient norm"""
    anneal_lr: bool = False
    """if toggled, the learning rate will be annealed"""

    # Network specific arguments
    encoder_type: str = "impala"
    """the type of the agent"""
    scale: int = 2
    """the width scale of the network"""
    pruning_type: str = "Baseline"
    """the pruning mode"""
    weight_decay: float = 0.0e-5
    """the weight decay for the optimizer"""
    latent_space_dim: int = 256
    """the latent space dimension"""
    cnn_filters: tuple = (16, 32, 32)
    """the number of filters for each CNN layer"""
    activation: str = 'relu'
    """the activation function of the network"""
    rescale_lr_by_scale: bool = True
    """if toggled, the learning rate will be rescaled by the width scale of the network"""

    # Network improvements
    use_pooling_layer: bool = False
    """if toggled, pooling layer will be enabled before the last linear layer"""
    pooling_layer_kernel_size: int = 1
    """the kernel size of the pooling layer"""
    use_dropout: bool = False
    """if toggled, dropout will be enabled"""
    use_1d_conv: bool = False
    """if toggled, 1D convolution will be enabled"""
    use_depthwise_conv: bool = False
    """if toggled, depthwise convolution will be enabled"""

    # ReDo settings
    redo_tau: float = 0.025
    """the tau for the ReDo algorithm"""
    redo_interval: int = 2000
    """the interval for the ReDo algorithm (computed in runtime)"""


if __name__ == "__main__":

    args = tyro.cli(Args)
    num_iterations = args.total_timesteps // args.num_envs // args.train_frequency
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    global progcen_hns
    if args.distribution_mode == "easy":
        progcen_hns.update(progcen_easy_hns)
    elif args.distribution_mode == "hard":
        progcen_hns.update(progcen_hard_hns)
    else:
        raise ValueError(f"Invalid distribution mode: {args.distribution_mode}")

    print(f"Run name: {run_name} | Batch size: {args.batch_size} | Num iterations: {num_iterations}")

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
        mode="disabled" if not args.track else "online",
    )
    wandb.run.log_code("./impoola")

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    if args.cuda:
        #     torch.backends.cudnn.allow_tf32 = True
        #     torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Device: {device}")

    # Environment that will be used for training
    envs = make_an_env(args, seed=args.seed,
                       normalize_reward=args.normalize_reward,
                       env_track_setting=args.env_track_setting,
                       full_distribution=False)

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Network creation
    q_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        use_pooling_layer=args.use_pooling_layer, pooling_layer_kernel_size=args.pooling_layer_kernel_size,
        use_dropout=args.use_dropout,
        use_1d_conv=args.use_1d_conv,
        use_depthwise_conv=args.use_depthwise_conv
    ).to(device)

    target_network = DQNAgent(
        encoder_type=args.encoder_type,
        envs=envs,
        width_scale=args.scale, out_features=args.latent_space_dim, cnn_filters=args.cnn_filters,
        activation=args.activation,
        use_layer_init_normed=False,
        use_pooling_layer=args.use_pooling_layer, pooling_layer_kernel_size=args.pooling_layer_kernel_size,
        use_dropout=args.use_dropout,
        use_1d_conv=args.use_1d_conv,
        use_depthwise_conv=args.use_depthwise_conv
    ).to(device)

    with torch.no_grad():
        example_input = 127 * np.ones((1,) + envs.single_observation_space.shape).astype(
            envs.single_observation_space.dtype)
        example_input = torch.tensor(example_input).to(device)
        q_network(example_input)
        target_network(example_input)

    # print summary of net
    statistics, total_params, m_macs, param_bytes = network_summary(q_network, example_input, device)

    # compile the model using torch
    if args.compile_agent:
        q_network = torch.compile(q_network, mode="reduce-overhead", fullgraph=True)
        target_network = torch.compile(target_network, mode="reduce-overhead", fullgraph=True)

    # benchmark the model
    if args.cuda and args.measure_latency:
        latency_256, latency_1 = measure_latency_agent(q_network, envs, device)
        print(f"Latency for a batch of 256 / 1: {latency_256:.2f} ms / {latency_1:.2f} ms")
        wandb.log({
            "charts/latency_256": latency_256,
            "charts/latency_1": latency_1,
        }, commit=False)

    wandb.log({
        "global_step": 0,
        "charts/total_network_params": total_params,
        "charts/total_network_m_macs": m_macs,
        "charts/total_network_param_bytes": param_bytes,
    })

    optimizer = optim.Adam(
        q_network.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,  # default eps=1e-8
        weight_decay=args.weight_decay,
        fused=True
    )

    if args.rescale_lr_by_scale:
        # LR was set for the default scale of 2, so we need to rescale it
        lr_scaling_factor = torch.tensor(args.scale / 2, device=device)
        optimizer.param_groups[0]['lr'].copy_(optimizer.param_groups[0]['lr'] / lr_scaling_factor)

    # copy the q_network
    target_network.load_state_dict(q_network.state_dict())

    # TRAINING STEP
    envs, q_network, global_step, b_obs = train_dqn_agent(args, envs, (q_network, target_network), optimizer, device)

    # Save statistics for reward normalization, will be used for fine-tuning on additional levels later
    if args.normalize_reward:
        train_envs_return_norm_mean = envs.return_rms.mean
        train_envs_return_norm_var = envs.return_rms.var
        train_envs_return_norm_count = envs.return_rms.count

    # Done!
    envs.close()
    agent = q_network

    # ANALYSIS OF THE FINAL MODEl
    save_agent_to_wandb(
        vars(args), agent, optimizer,
        envs.obs_rms if hasattr(envs, "obs_rms") else None,
        envs.return_rms if hasattr(envs, "return_rms") else None,
        metadata={
            "global_step": global_step,
        }, aliases=["latest"])

    # Check how much GPU memory is used. If less than 5 GB are available, run the redo algorithm on the CPU
    if torch.cuda.get_device_properties(0).total_memory < 5e9:
        agent = agent.to('cpu')
        b_obs = b_obs.to('cpu')
        print("Running ReDo on CPU")

    redo_dict = run_redo(b_obs[:32], agent, optimizer, args.redo_tau, False, False)
    flatten_dict_list_redo_per_layer = {f"dormant_neurons/{i}_{k}": v for i, (k, v) in
                                        enumerate(redo_dict['dormant_neurons_per_layer'].items())}

    wandb.log({
        "global_step": global_step,
        "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
        "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
        **flatten_dict_list_redo_per_layer
    })
    agent = agent.to(device)
    print(f"Zero fraction: {redo_dict['zero_fraction']:.2f} | Dormant fraction: {redo_dict['dormant_fraction']:.2f}")

    if envs.env_type == "atari":
        exit(0)

    # EVALUATION
    print("Running evaluation!")
    eval_args = deepcopy(args)

    # EVALUATION TRACK (1): In-distribution generalization only for generalization track
    evaluation.run_training_track(agent, eval_args, global_step)

    # EVALUATION TRACK (2): Out-of-distribution generalization for full distribution
    if args.env_track_setting == "generalization":
        evaluation.run_test_track(agent, eval_args, global_step)
