# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
from collections import deque

import wandb
import numpy as np
import torch

from impoola.maker.make_env import progcen_hns, atari_games_list, make_an_env
import time


def rollout(envs, agent, n_episodes=10000, noise_scale=None, deterministic=True):
    device = next(agent.parameters()).device

    # We cannot simply append eps whenever one is ready because this would bias the evaluation towards eps that are fast
    eval_avg_return = []
    eps_to_do_per_env = np.zeros(envs.num_envs)
    for idx in range(n_episodes):
        eps_to_do_per_env[idx % envs.num_envs] += 1

    assert sum(eps_to_do_per_env) == n_episodes, f"Sum of eps_to_do_per_env is broken: {sum(eps_to_do_per_env)}"

    agent.eval()
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    obs_shape = next_obs.shape

    with torch.inference_mode():  # TODO: Can it be that this here influences the layer norm?
        while len(eval_avg_return) < n_episodes:
            action = agent.get_action(next_obs, deterministic=deterministic)
            next_obs, _, terminated, truncated, info = envs.step(action.cpu().numpy())

            if noise_scale is not None:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                noise = torch.randn(obs_shape, device=device) * noise_scale
                next_obs.add_(noise.round()).clamp_(0.0, 255.0)
            else:
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)

            if envs.env_type == "atari":
                next_done = np.logical_or(terminated, truncated)
                for idx, d in enumerate(next_done):
                    if d and info["lives"][idx] == 0:
                        eval_avg_return.append(info["r"][idx])
                        eps_to_do_per_env[idx] -= 1
            else:
                if "_episode" in info.keys():
                    for i in range(len(info["_episode"])):
                        if info["_episode"][i] and eps_to_do_per_env[i] > 0:
                            eval_avg_return.append(info["episode"]["r"][i])
                            eps_to_do_per_env[i] -= 1
            # ikk += 1
            # print(ikk)
            # if ikk > 55:
            #     import matplotlib.pyplot as plt
            #     plt.imshow(next_obs[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
            #     # save the image
            #     # remove frame for the iamge
            #     plt.axis('off')
            #     plt.savefig(f"image_{ikk}.png")
            #     exit(0)
            #     plt.show()
    agent.train()
    return eval_avg_return


def _get_normalized_score(eval_avg_return, game_range):
    if game_range is not None:
        normalized_score = (np.mean(eval_avg_return) - game_range[1]) / (game_range[2] - game_range[1])
    else:
        normalized_score = np.mean(eval_avg_return)
    return normalized_score


def _get_game_range(env_id):
    for game_name, game_range in progcen_hns.items():
        if env_id in game_name:
            print(f"Game range: {game_range}")
            return game_range
    if env_id in atari_games_list:
        return None
    raise ValueError(f"Unknown environment: {env_id}")


def get_normalized_score(env_id, eval_avg_return):
    game_range = _get_game_range(env_id)
    return _get_normalized_score(eval_avg_return, game_range)


def _evaluate_and_log_results(env_id, eval_avg_return, global_step, prefix, postfix=""):
    normalized_score = get_normalized_score(env_id, eval_avg_return)
    wandb.log({
        f"global_step{postfix}": global_step,
        f"scores{postfix}/normalized_score_{prefix}": normalized_score,
        f"scores{postfix}/eval_avg_return_{prefix}": np.mean(eval_avg_return),
    })
    print(f"\nNormalized score {prefix} ({global_step}): {normalized_score}")


def run_training_track(agent, args, global_step=None, postfix=""):
    # print("\nEvaluation: Training Track")
    # envs = make_procgen_env(
    #     args,
    #     args.env_track_setting,
    #     full_distribution=False,
    #     normalize_reward=False,
    #     rand_seed=args.seed,
    #     distribution_mode=args.distribution_mode
    # )
    envs = make_an_env(args, seed=args.seed, normalize_reward=False,
                       env_track_setting=args.env_track_setting, full_distribution=False)

    eval_avg_return = rollout(envs, agent, args.n_episodes_rollout, deterministic=args.deterministic_rollout)
    envs.close()
    _evaluate_and_log_results(args.env_id, eval_avg_return, global_step, "train", postfix)


def run_test_track(agent, args, global_step=None, postfix=""):
    # # print("\nEvaluation: Test Track")
    # envs = make_procgen_env(
    #     args,
    #     args.env_track_setting,
    #     full_distribution=True,
    #     normalize_reward=False,
    #     rand_seed=args.seed,
    #     render=False,
    #     distribution_mode=distribution_mode
    # )
    envs = make_an_env(args, seed=args.seed, normalize_reward=False,
                       env_track_setting=args.env_track_setting, full_distribution=True)

    eval_avg_return_test = rollout(envs, agent, args.n_episodes_rollout, deterministic=args.deterministic_rollout)
    envs.close()
    _evaluate_and_log_results(args.env_id, eval_avg_return_test, global_step, "test", postfix)


def run_noise_robustness_track(agent, args, noise_scales=(5, 15, 30), global_step=None, postfix="",
                               distribution_mode="easy"):
    # print("\nEvaluation: Noise Robustness Track")
    for noise_scale in noise_scales:
        # envs = make_procgen_env(
        #     args,
        #     args.env_track_setting,
        #     full_distribution=True,
        #     normalize_reward=False,
        #     rand_seed=args.seed,
        #     distribution_mode=distribution_mode
        # )
        envs = make_an_env(args, seed=args.seed, normalize_reward=False,
                           env_track_setting=args.env_track_setting, full_distribution=True)

        eval_avg_return_noise = rollout(envs, agent, args.n_episodes_rollout, noise_scale, deterministic=args.deterministic_rollout)
        envs.close()
        _evaluate_and_log_results(args.env_id, eval_avg_return_noise, global_step, f"noise_{noise_scale}", postfix)


def run_fine_tuning_track(agent, args, global_step=None, before=False, postfix="", distribution_mode="easy"):
    # print(f"\nEvaluation: Fine Tuning Track {'(Before)' if before else ''}")
    # envs = make_procgen_env(
    #     args,
    #     "fine_tuning",
    #     full_distribution=False,
    #     normalize_reward=False,
    #     rand_seed=args.seed,
    #     distribution_mode=distribution_mode
    # )
    envs = make_an_env(args, seed=args.seed, normalize_reward=False,
                       env_track_setting="fine_tuning", full_distribution=False)

    eval_avg_return_fine_tuning = rollout(envs, agent, args.n_episodes_rollout, deterministic=args.deterministic_rollout)
    envs.close()
    prefix = "fine_tuning" + ("_before" if before else "")
    _evaluate_and_log_results(args.env_id, eval_avg_return_fine_tuning, global_step, prefix, postfix)
