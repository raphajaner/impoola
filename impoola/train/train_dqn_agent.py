import random
import time
from collections import deque
from copy import deepcopy

import wandb
import numpy as np
import torch
from tqdm import trange
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer

from impoola.utils.utils import StopTimer
from impoola.utils.schedules import linear_schedule
from impoola.eval.evaluation import run_test_track, run_training_track, _get_normalized_score, _get_game_range
from impoola.prune.redo import run_redo


def make_replay_buffer(args, envs, device):
    if args.prioritized_replay:
        from impoola.utils.replay_buffer import SimplifiedPrioritizedMultiStepReplayBuffer as Buffer
        # from impoola.utils.replay_buffer import PrioritizedMultiStepReplayBuffer as Buffer
        replay_buffer = Buffer(
            args.buffer_size,
            envs.single_observation_space_gymnasium,
            envs.single_action_space_gymnasium,
            device,
            optimize_memory_usage=False,
            handle_timeout_termination=False,
            n_envs=envs.num_envs,
            n_steps=args.multi_step,
            gamma=args.gamma,
            alpha=0.5,
            beta=0.5,
            beta_increment_per_sampling=0  # (1 - 0.4) / (args.total_timesteps // args.num_envs),
        )
    else:
        if args.multi_step > 1:
            from impoola.utils.replay_buffer import MultiStepReplayBuffer as Buffer
            replay_buffer = Buffer(
                args.buffer_size,
                envs.single_observation_space_gymnasium,
                envs.single_action_space_gymnasium,
                device,
                optimize_memory_usage=False,
                handle_timeout_termination=False,
                n_envs=envs.num_envs,
                n_steps=args.multi_step,
                gamma=args.gamma,
            )
        else:
            replay_buffer = ReplayBuffer(
                args.buffer_size,
                envs.single_observation_space_gymnasium,
                envs.single_action_space_gymnasium,
                device,
                optimize_memory_usage=False,
                handle_timeout_termination=False,
                n_envs=envs.num_envs,
            )
    return replay_buffer


def train_dqn_agent(args, envs, agent, optimizer, device):
    """ Train the DQN agent """
    q_network, target_network = agent
    game_range = _get_game_range(args.env_id)

    if args.prioritized_replay:
        def unwrap_data(data):
            data, replay_indices, replay_weights = data
            return data, replay_indices, replay_weights
    else:
        def unwrap_data(data):
            return data, None, torch.ones(data.rewards.shape, device=device)

    if args.double_dqn:
        def get_target_max(target_network, next_observations, actions):
            online_actions = q_network(next_observations).argmax(dim=1, keepdim=True)
            target_max = target_network(next_observations)
            target_max = torch.gather(target_max, 1, online_actions).squeeze()
            return target_max
    else:
        def get_target_max(target_network, next_observations, actions):
            target_max = target_network(next_observations).max(dim=1)
            return target_max

    def update_dqn(data, args, q_network, target_network, optimizer):
        with torch.no_grad():
            data, replay_indices, replay_weights = unwrap_data(data)

            # if args.double_dqn:
            online_actions = q_network(data.next_observations).argmax(dim=1, keepdim=True)
            target_max = target_network(data.next_observations)
            target_max = torch.gather(target_max, 1, online_actions).squeeze()
            # else:
            #     target_max, _ = target_network(data.next_observations).max(dim=1)
            # target_max = get_target_max(target_network, data.next_observations, data.actions)
            td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())

        old_val = q_network(data.observations)
        old_val = torch.gather(old_val, 1, data.actions).squeeze()
        assert td_target.shape == old_val.shape, f"Shapes of td_target and old_val do not match: {td_target.shape} vs {old_val.shape}"

        td_error = td_target - old_val
        td_error = td_error * replay_weights
        loss = torch.mean(td_error ** 2)

        # optimize the model
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
        optimizer.step()

        return loss, old_val, td_error, replay_weights, replay_indices

    if args.compile_agent:
        update_dqn = torch.compile(update_dqn, mode="reduce-overhead")

    replay_buffer = make_replay_buffer(args, envs, device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_eval_step = -1  # Set to -1 to make sure to not evaluate at the very last step
    bar = trange(args.total_timesteps, initial=0, position=0)

    avg_returns = deque(maxlen=20)
    avg_sps = deque(maxlen=100)

    obs, _ = envs.reset()

    # Initial dormant neurons
    redo_dict = run_redo(torch.tensor(obs[:32], device=device), q_network, optimizer, args.redo_tau, False, False)

    flatten_dict_list_redo_per_layer = {f"dormant_neurons/{i}_{k}": v for i, (k, v) in
                                        enumerate(redo_dict['dormant_neurons_per_layer'].items())}

    wandb.log({
        "global_step": global_step,
        "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
        "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
        **flatten_dict_list_redo_per_layer
    })

    duration_linear_schedule = args.exploration_fraction * args.total_timesteps

    stop_timer = StopTimer()
    stop_timer.start()

    while global_step < args.total_timesteps:
        start_time = time.time()

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, duration_linear_schedule, global_step)

        if args.anneal_lr:
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        # Exploration
        if args.softmax_exploration:
            with torch.inference_mode():
                q_values = q_network(torch.tensor(obs, device=device))
                temperature = 1.0
                action_probs = F.softmax(q_values / temperature, dim=1)
                actions = torch.multinomial(action_probs, num_samples=1).squeeze().cpu().numpy()
        else:
            if random.random() < epsilon:
                # TODO: we could sample for each environment separately
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                with torch.no_grad():  # with torch.inference_mode():  <- issues with torch.compile
                    q_values = q_network(torch.tensor(obs, device=device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, info = envs.step(actions)
        next_done = np.logical_or(terminated, truncated)

        # update the bar by the taken step taking the number of environments into account
        global_step += envs.num_envs
        bar.n = global_step
        bar.refresh()

        if envs.env_type == "atari":
            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    wandb.log({
                        f"global_step": global_step,
                        f"charts/episodic_return": info["r"][idx],
                        f"charts/avg_episodic_return": np.average(avg_returns),
                        f"charts/episodic_length": info["l"][idx],
                    })
        else:
            if "_episode" in info.keys():
                wandb.log({
                    f"global_step": global_step,
                    f"charts/episodic_return": np.mean(info["episode"]["r"][info["_episode"]]),
                    f"charts/episodic_return_normalized": _get_normalized_score(
                        info["episode"]["r"][info["_episode"]], game_range),
                    f"charts/episodic_length": np.mean(info["episode"]["l"][info["_episode"]]),
                })

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            if global_step % args.train_frequency == 0:

                # if train_frequency is smaller than one, we want to update more than once per step
                num_updates = int(1 / args.train_frequency) if args.train_frequency < 1 else int(args.train_frequency)

                for _ in range(num_updates):
                    data = replay_buffer.sample(args.batch_size)
                    loss, old_val, td_error, replay_weights, replay_indices = \
                        update_dqn(data, args, q_network, target_network, optimizer)

                    if args.prioritized_replay:
                        replay_buffer.update_priorities(replay_indices, td_error.detach().abs().cpu().numpy())
                        assert replay_weights.shape == td_error.shape, f"Replay_weights and td_error do not match"

                wandb.log({
                    f"global_step": global_step,
                    f"losses/td_loss": loss,
                    f"losses/q_values": old_val.mean().item(),
                    f"charts/learning_rate": optimizer.param_groups[0]["lr"],
                    f"charts/epsilon": epsilon,
                    f"charts/replay_buffer_beta": replay_buffer.beta if args.prioritized_replay else 0.0,
                })

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_network_state_dict = q_network.state_dict()
                if args.tau is None:
                    target_network.load_state_dict(q_network_state_dict)
                else:
                    target_state_dict = target_network.state_dict()
                    for key, value in target_state_dict.items():
                        target_state_dict[key] = \
                            args.tau * q_network_state_dict[key] + (1.0 - args.tau) * target_state_dict[key]
                    target_network.load_state_dict(target_state_dict)

        replay_buffer.add(obs, next_obs, actions, rewards, terminated, info)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        avg_sps.append(envs.num_envs / (time.time() - start_time))
        wandb.log({
            f"global_step": global_step,
            f"charts/avg_sps": int(np.mean(avg_sps))
        })

        if global_step >= last_eval_step + args.training_eval_ratio * args.total_timesteps:
            stop_timer.stop()

            # Estimate number of dormant neurons
            redo_dict = run_redo(torch.tensor(obs[:32], device=device), q_network, optimizer, args.redo_tau, False,
                                 False)

            flatten_dict_list_redo_per_layer = {f"dormant_neurons/{i}_{k}": v for i, (k, v) in
                                                enumerate(redo_dict['dormant_neurons_per_layer'].items())}

            wandb.log({
                "global_step": global_step,
                "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
                "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
                **flatten_dict_list_redo_per_layer
            })

            # calc_translation_sensitivity(q_network, torch.tensor(obs[:32], device=device), device)

            # Do not evaluate on Atari environments but show training progress
            if envs.env_type != "atari":
                eval_args = deepcopy(args)
                eval_args.n_episodes_rollout = int(1e3)
                run_training_track(q_network, eval_args, global_step)
                if args.env_track_setting == "generalization":
                    run_test_track(q_network, eval_args, global_step)

            last_eval_step = global_step
            stop_timer.start()

    wandb.log({
        "global_step": global_step,
        f"charts/avg_sps": int(np.mean(avg_sps)),
        f"charts/elapsed_train_time": stop_timer.get_elapsed_time(),
    })
    # Free memory of replay buffer
    del replay_buffer
    return envs, q_network, global_step, torch.tensor(obs, device=device)
