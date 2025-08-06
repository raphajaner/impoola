import time
from collections import deque
from typing import Optional

import numpy as np
import gym


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, _ = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = super().step(action)
        dones = np.logical_or(terminateds, truncateds)

        self.episode_returns += infos["reward"] if self.atari else rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        self.episode_returns *= 1 - infos["terminated"] if self.atari else 1 - terminateds
        self.episode_lengths *= 1 - infos["terminated"] if self.atari else 1 - terminateds
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class EpisodicLifeRecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations, _ = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, terminateds, truncateds, infos = super().step(action)
        dones = np.logical_or(terminateds, truncateds)

        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths

        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return observations, rewards, dones, infos,


class ProcgenRecordEpisodeStatistics(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.

    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.

    After the completion of an episode, ``info`` will look like this::

        >>> info = {
        ...     ...
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since instantiation of wrapper>"
        ...     },
        ... }

    For a vectorized environments the output will be in the form of::

        >>> infos = {
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since instantiation of wrapper>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }

    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.

    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        """This wrapper will keep track of cumulative rewards and episode lengths.

        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        dones = np.logical_or(terminateds, truncateds)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }

                infos = add_vector_episode_statistics(
                    infos, episode_info["episode"], self.num_envs, i
                )

                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def add_vector_episode_statistics(
        info: dict, episode_info: dict, num_envs: int, env_num: int
):
    """Add episode statistics.

    Add statistics coming from the vectorized environment.

    Args:
        info (dict): info dict of the environment.
        episode_info (dict): episode statistics data.
        num_envs (int): number of environments.
        env_num (int): env number of the vectorized environments.

    Returns:
        info (dict): the input info dict with the episode statistics.
    """
    info["episode"] = info.get("episode", {})

    info["_episode"] = info.get("_episode", np.zeros(num_envs, dtype=bool))
    info["_episode"][env_num] = True

    for k in episode_info.keys():
        info_array = info["episode"].get(k, np.zeros(num_envs))
        info_array[env_num] = episode_info[k]
        info["episode"][k] = info_array

    return info



# class NoisyPixelObservations(gym.ObservationWrapper):
#     def __init__(self, env, obs_shape, noise_scale=0.01):
#         super().__init__(
#             env,
#         )
#         self.obs_shape = obs_shape
#         self.noise_scale = noise_scale
#
#     def observation(self, observation):
#         return np.clip(observation + np.round(np.random.normal(loc=0, scale=self.noise_scale, size=self.obs_shape)), 0,
#                        255).astype(observation.dtype)
# import pdb; pdb.set_trace()
# # Noise is added per pixel, quantized to integers
# noise = np.random.normal(loc=0, scale=self.noise_scale, size=observation.shape)
# return np.clip(observation + np.round(noise).astype(int), 0, 255)
