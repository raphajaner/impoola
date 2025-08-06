# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import time
from collections import deque
from copy import deepcopy

import wandb
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange

from impoola.prune.redo import run_redo
from impoola.eval.evaluation import run_test_track, run_training_track, _get_normalized_score, _get_game_range
from impoola.utils.utils import StopTimer


def train_ppo_agent(args, envs, agent, optimizer, device):
    """ Train the PPO agent """
    postfix = ""
    game_range = _get_game_range(args.env_id)

    avg_returns = deque(maxlen=20)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.bool)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    logits = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n), device=device)

    gamma = torch.tensor(args.gamma, device=device)
    gae_lambda = torch.tensor(args.gae_lambda, device=device)
    clip_coef = torch.tensor(args.clip_coef, device=device)
    norm_adv = torch.tensor(args.norm_adv, device=device)
    ent_coef = torch.tensor(args.ent_coef, device=device)
    vf_coef = torch.tensor(args.vf_coef, device=device)
    clip_vloss = torch.tensor(args.clip_vloss, device=device)
    learning_rate = optimizer.param_groups[0]["lr"].clone()
    max_grad_norm = torch.tensor(args.max_grad_norm, device=device)

    from impoola.train.ppo_criterion import ppo_loss, ppo_gae

    if args.compile_agent:
        ppo_loss = torch.compile(ppo_loss, mode="reduce-overhead", fullgraph=True)
        ppo_gae = torch.compile(ppo_gae, mode="reduce-overhead", fullgraph=True)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    last_eval_step = -1  # Set to -1 to make sure to not evaluate at the very last step
    pruning_steps_done = 0

    avg_sps = deque(maxlen=100)
    stop_timer = StopTimer()
    stop_timer.start()

    next_obs, _ = envs.reset()

    next_obs = torch.tensor(next_obs, device=device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)

    # Initial dormant neurons
    if not args.use_moe:
        redo_dict = run_redo(next_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

        flatten_dict_list_redo_per_layer = {f"dormant_neurons/{i}_{k}": v for i, (k, v) in
                                            enumerate(redo_dict['dormant_neurons_per_layer'].items())}

        wandb.log({
            "global_step": global_step,
            "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
            "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
            **flatten_dict_list_redo_per_layer
        })

    if args.pruning_type != "Baseline":
        from impoola.maker.make_pruner import make_pruner
        from impoola.prune.pruning_func import pruning_step
        from impoola.utils.utils import calculate_global_parameters_number

        pruner, pruning_func, zero_weight_mode = make_pruner(args, agent, args.num_iterations)
        base_network_params = current_network_params = \
            calculate_global_parameters_number(agent, zero_weight_mode=zero_weight_mode)
        global_sparsity = 0

    for iteration in trange(1, args.num_iterations + 1):
        start_time = time.time()

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * learning_rate
            # optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, pi_logits = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                logits[step] = pi_logits
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminated, truncated)

            rewards[step] = torch.tensor(reward, device=device).view(-1)
            next_obs = torch.tensor(next_obs, device=device)
            next_done = torch.tensor(next_done, device=device, dtype=torch.bool)
            if envs.env_type == "atari":
                for idx, d in enumerate(next_done):
                    if d and info["lives"][idx] == 0:
                        avg_returns.append(info["r"][idx])
                        wandb.log({
                            f"global_step{postfix}": global_step,
                            f"charts{postfix}/episodic_return": info["r"][idx],
                            f"charts{postfix}/avg_episodic_return": np.average(avg_returns),
                            f"charts{postfix}/episodic_length": info["l"][idx],
                        })
            else:
                if "_episode" in info.keys():
                    wandb.log({
                        f"global_step{postfix}": global_step,
                        f"charts{postfix}/episodic_return": np.mean(info["episode"]["r"][info["_episode"]]),
                        f"charts{postfix}/episodic_return_normalized": _get_normalized_score(
                            info["episode"]["r"][info["_episode"]], game_range),
                        f"charts{postfix}/episodic_length": np.mean(info["episode"]["l"][info["_episode"]]),
                    })

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_values = values.reshape(-1)

        # advantages, returns = ppo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device,
        #                                args.num_steps)

        advantages, returns = ppo_gae(agent, next_done, next_obs, rewards, dones, values, gamma, gae_lambda, device,
                                      args.num_steps)

        # if not torch.allclose(advantages, advantages2):
        #     print("advantages do not match")
        #     import pdb; pdb.set_trace()
        #
        # if not torch.allclose(returns, returns2):
        #     print("returns do not match")
        #     import pdb; pdb.set_trace()
        # returns = advantages + values  # returns2 + 1e-10 #.clone()
        advantages = advantages.clone()
        returns = returns.clone()

        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []

        # agent.train()
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(args.batch_size, device=device)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                loss, pg_loss, v_loss, entropy_loss, logratio, ratio = ppo_loss(
                    agent,
                    b_obs[mb_inds],
                    b_logprobs[mb_inds], b_actions[mb_inds],
                    b_values[mb_inds], b_returns[mb_inds],
                    b_advantages[mb_inds],
                    b_advantages[mb_inds],
                    norm_adv, clip_coef, ent_coef, vf_coef, clip_vloss
                )

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        avg_sps.append(envs.num_envs / (time.time() - start_time))
        wandb.log({
            f"global_step": global_step,
            f"charts/avg_sps": int(np.mean(avg_sps))
        })

        if global_step >= last_eval_step + args.training_eval_ratio * args.total_timesteps:
            stop_timer.stop()

            # Estimate number of dormant neurons
            if not args.use_moe:
                redo_dict = run_redo(b_obs[:args.minibatch_size], agent, optimizer, args.redo_tau, False, False)

                flatten_dict_list_redo_per_layer = {f"dormant_neurons/{i}_{k}": v for i, (k, v) in
                                                    enumerate(redo_dict['dormant_neurons_per_layer'].items())}

                wandb.log({
                    "global_step": global_step,
                    "dormant_neurons/zero_fraction": redo_dict['zero_fraction'],
                    "dormant_neurons/dormant_fraction": redo_dict['dormant_fraction'],
                    **flatten_dict_list_redo_per_layer
                })

            # Do not evaluate on Atari environments but show training progress
            if envs.env_type != "atari":
                eval_args = deepcopy(args)
                eval_args.n_episodes_rollout = int(1e3)
                run_training_track(agent, eval_args, global_step)
                if args.env_track_setting == "generalization":
                    run_test_track(agent, eval_args, global_step)

            last_eval_step = global_step
            stop_timer.start()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.pruning_type == "UnstructuredNorm":
            did_prune, current_network_params, global_sparsity = pruning_step(
                args, agent, optimizer, pruning_func, pruner, iteration,
                zero_weight_mode, base_network_params, current_network_params, global_sparsity,
                b_obs,
            )

        wandb.log({
            f"global_step{postfix}": global_step,
            f"losses{postfix}/value_loss": v_loss.item(),
            f"losses{postfix}/policy_loss": pg_loss.item(),
            f"losses{postfix}/entropy": entropy_loss.item(),
            f"losses{postfix}/old_approx_kl": old_approx_kl.item(),
            f"losses{postfix}/approx_kl": approx_kl.item(),
            f"losses{postfix}/clipfrac": np.mean(clipfracs),
            f"losses{postfix}/explained_variance": explained_var,
            f"charts{postfix}/learning_rate": optimizer.param_groups[0]["lr"],
            f"charts{postfix}/pruning_steps_done": pruning_steps_done,
            f"charts{postfix}/avg_sps": int(np.mean(avg_sps)),
            f"charts{postfix}/elapsed_train_time": stop_timer.get_elapsed_time(),
        })

    return envs, agent, global_step, b_obs
