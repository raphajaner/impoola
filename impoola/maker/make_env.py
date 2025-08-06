# import envpool
import numpy as np
from procgen.env import ProcgenGym3Env, ToBaselinesVecEnv
import gym
from gym3 import ViewerWrapper
import gymnasium

from impoola.eval.normalized_score_lists import progcen_hns


atari_games_list = [
    "MsPacman-v5",
    "Pong-v5",
    "Qbert-v5",
    "Assault-v5",
    "Asterix-v5",
    "BeamRider-v5",
    "Boxing-v5",
    "Breakout-v5",
    "CrazyClimber-v5",
    "DemonAttack-v5",
    "Enduro-v5",
    "FishingDerby-v5",
    "SpaceInvaders-v5",
    "Tutankham-v5",
    "VideoPinball-v5",
    "Seaquest-v5"
]

procgen_games_easy_list = [
    "BigfishEasy-v0",
    "BossfightEasy-v0",
    "CaveflyerEasy-v0",
    "ChaserEasy-v0",
    "CliffwalkerEasy-v0",
    "CoinrunEasy-v0",
    "DodgeballEasy-v0",
    "FruitbotEasy-v0",
    "HeistEasy-v0",
    "JumperEasy-v0",
    "LeaperEasy-v0",
    "MazeEasy-v0",
    "MinerEasy-v0",
    "NinjaEasy-v0",
    "PlunderEasy-v0",
    "StarpilotEasy-v0",
]

procgen_games_hard_list = [
    "BigfishHard-v0",
    "BossfightHard-v0",
    "CaveflyerHard-v0",
    "ChaserHard-v0",
    "CliffwalkerHard-v0",
    "CoinrunHard-v0",
    "DodgeballHard-v0",
    "FruitbotHard-v0",
    "HeistHard-v0",
    "JumperHard-v0",
    "LeaperHard-v0",
    "MazeHard-v0",
    "MinerHard-v0",
    "NinjaHard-v0",
    "PlunderHard-v0",
    "StarpilotHard-v0",
]

mujoco_games_list = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]


class ProcgenToGymNewAPI(gym.Wrapper):
    def reset(self, **kwargs):
        return super().reset(**kwargs), {}

    def step(self, action):
        ob, reward, done, info = super().step(action)
        # combine list of infos into one dict
        dict_info = {}
        for key in info[0].keys():
            dict_info[key] = [inf[key] for inf in info]
        terminated = done
        truncated = np.full_like(done, False, dtype=bool)
        return ob, reward, terminated, truncated, dict_info


def _make_procgen_env(num_envs, env_id, env_track, num_levels, rand_seed, render=False, distribution_mode="easy"):
    # print(f"Using track: {env_track} (num_levels: {num_levels})")

    # if distribution_mode != "easy" and env_id not in ["coinrun", "ninja", "climber", "fruitbot", "caveflyer"]:
    #     raise ValueError(f"Only coinrun can be used with distribution_mode: {distribution_mode}")

    envs = ProcgenGym3Env(
        num=num_envs,
        env_name=env_id,
        num_levels=num_levels,
        start_level=(201 if distribution_mode == "easy" else 501) if env_track == "fine_tuning" else 0,
        # Note: Start_level has no influence when num_levels=0
        distribution_mode=distribution_mode,
        rand_seed=rand_seed,
        render_mode="rgb_array" if render else None,
    )

    if render:
        envs = ViewerWrapper(envs, info_key="rgb")
    envs = ProcgenToGymNewAPI(ToBaselinesVecEnv(envs))

    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs["rgb"].transpose((0, 3, 1, 2)))
    envs.single_action_space = envs.action_space
    envs.single_observation_space = type(envs.observation_space["rgb"])(low=0, high=255, shape=(3, 64, 64),
                                                                        dtype=np.uint8)
    # TODO: Fix that only gym is used!
    envs.single_observation_space_gymnasium = gymnasium.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    envs.single_action_space_gymnasium = gymnasium.spaces.Discrete(envs.single_action_space.n)

    envs.is_vector_env = True
    envs.env_type = 'procgen'
    return envs


def make_procgen_env(args, env_track, full_distribution=False, normalize_reward=False, rand_seed=None, render=False,
                     distribution_mode="easy"):
    # Num levels is 200 for easy games and 1000 for hard games when running in general track
    if env_track == "generalization":
        num_levels = 200 if distribution_mode == "easy" else 500
    elif env_track == "efficiency":
        num_levels = 0
    elif env_track == "fine_tuning":
        num_levels = 100
    else:
        raise ValueError(f"Invalid env_track: {env_track}")

    if full_distribution and env_track == "fine_tuning":
        raise ValueError("Cannot use full distribution in fine tuning track")

    num_levels = 0 if full_distribution else num_levels
    envs = _make_procgen_env(args.num_envs, args.env_id, env_track, num_levels, rand_seed, render, distribution_mode)

    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        run_name = f"{args.env_id}_{args.seed}"
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    if normalize_reward:
        envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


class EnvpoolToGymNewAPI(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        # set is_vector_env to True
        setattr(self, "is_vector_env", True)
        setattr(self, "num_envs", env.spec.config.num_envs)
        setattr(self, "single_action_space", env.action_space)
        setattr(self, "single_observation_space", env.observation_space)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def step(self, action):
        ob, reward, terminated, truncated, dict_info = super().step(action)

        return ob, reward, terminated, truncated, dict_info


class RecordEpisodeStatistics2(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        )


def make_an_env(args, seed, normalize_reward, env_track_setting="generalization", full_distribution=False):
    if args.env_id in atari_games_list:
        envs = make_atari_env(
            args,
            env_id=args.env_id,
            num_envs=args.num_envs,
            seed=seed,
            normalize_reward=normalize_reward
        )
    elif args.env_id in progcen_hns.keys():
        envs = make_procgen_env(
            args,
            env_track_setting,
            full_distribution=full_distribution,
            normalize_reward=normalize_reward,
            rand_seed=seed,
            distribution_mode=args.distribution_mode
        )
    else:
        raise ValueError(f"Unknown environment: {args.env_id}")
    return envs


def make_atari_env(args, env_id, num_envs, seed, normalize_reward=False):
    import envpool
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=seed,
    )
    envs.env_type = 'atari'

    envs = EnvpoolToGymNewAPI(envs)
    envs = RecordEpisodeStatistics2(envs)
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if args.capture_video:
        run_name = f"{args.env_id}_{args.seed}"
        envs = gym.wrappers.RecordVideo(envs, f"videos/{run_name}")

    if normalize_reward:
        envs = gym.wrappers.NormalizeReward(envs, gamma=args.gamma)
        envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs

# def make_envpool_env(args, eval_mode=False):
#     if args.env_id in atari_games_list:
#         envs = envpool.make(
#             args.env_id,
#             env_type="gym",
#             num_envs=args.num_envs,
#             episodic_life=True,
#             reward_clip=True,
#             seed=args.seed,
#         )
#         print(f"Using atari env: {args.env_id}")
#         envs = EpisodicLifeRecordEpisodeStatistics(envs)
#         envs.env_type = 'atari'
#         return envs
#
#     elif args.env_id in procgen_games_easy_list or args.env_id in procgen_games_hard_list:
#         if not eval_mode:
#             num_levels = 200 if args.env_id in procgen_games_easy_list else 500
#             start_level = 0
#         else:
#             num_levels = 0
#             start_level = 0
#
#         envs = envpool.make(
#             args.env_id,
#             env_type="gym",
#             num_envs=args.num_envs,
#             seed=args.seed,
#             num_levels=num_levels,
#             start_level=start_level,
#             use_sequential_levels=False,
#         )
#         envs = EnvpoolToGymNewAPI(envs)
#         envs = gym.wrappers.RecordEpisodeStatistics(envs)
#         envs.env_type = 'procgen'
#         print(f"Using procgen env: {args.env_id}")
#     else:
#         envs = envpool.make(
#             args.env_id,
#             env_type="gym",
#             num_envs=args.num_envs,
#             seed=args.seed,
#         )
#         envs.env_type = 'unknown'
#         envs = RecordEpisodeStatistics(envs)
#         print(f"Using unknown env: {args.env_id}")
#
#     return envs
