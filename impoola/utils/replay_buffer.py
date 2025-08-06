import sys

sys.setrecursionlimit(10000)

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples
import torch


# from stable_baselines3.common.segment_tree import SumSegmentTree, MinSegmentTree

class MultiStepReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, optimize_memory_usage=False,
                 handle_timeout_termination=False, n_envs=1, n_steps=3, gamma=0.99):
        # Initialize the ReplayBuffer with a single environment's observation and action spaces
        assert not optimize_memory_usage, "Memory optimization is not supported for multi-step replay buffers"
        super(MultiStepReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, 1,
                                                    optimize_memory_usage, handle_timeout_termination)
        self.n_envs_ = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffers = [[] for _ in range(n_envs)]

    def add(self, obs, next_obs, action, reward, done, infos):
        for env_idx in range(self.n_envs_):
            self.n_step_buffers[env_idx].append(
                (obs[env_idx], next_obs[env_idx], action[env_idx], reward[env_idx], done[env_idx]))

            if len(self.n_step_buffers[env_idx]) >= self.n_steps or done[env_idx]:
                discounted_reward = 0
                for i in range(len(self.n_step_buffers[env_idx])):
                    discounted_reward += self.n_step_buffers[env_idx][i][3] * (self.gamma ** i)
                    if self.n_step_buffers[env_idx][i][4]:  # If done, break early
                        break

                obs_, next_obs_, action_ = self.n_step_buffers[env_idx][0][:3]

                super(MultiStepReplayBuffer, self).add(
                    np.expand_dims(obs_, axis=0),
                    np.expand_dims(next_obs_, axis=0),
                    np.expand_dims(action_, axis=0),
                    np.expand_dims(discounted_reward, axis=0),
                    np.expand_dims(done[env_idx], axis=0),
                    []  # [infos[env_idx]]
                )

                self.n_step_buffers[env_idx].pop(0)

            if done[env_idx]:
                self.n_step_buffers[env_idx] = []


class PrioritizedMultiStepReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, optimize_memory_usage=False,
                 handle_timeout_termination=False, n_envs=1, n_steps=3, gamma=0.99, alpha=0.6, beta=0.4,
                 beta_increment_per_sampling=0.001):
        assert not optimize_memory_usage, "Memory optimization is not supported for multi-step replay buffers"
        super(PrioritizedMultiStepReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, 1,
                                                               optimize_memory_usage, handle_timeout_termination)
        self.n_envs_ = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_step_buffers = [[] for _ in range(n_envs)]
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = 1.0
        self.epsilon = 1e-6

        # Prioritized replay buffer structures
        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self.sum_tree = SumSegmentTree(it_capacity)
        self.min_tree = MinSegmentTree(it_capacity)

    def add(self, obs, next_obs, action, reward, done, infos):
        for env_idx in range(self.n_envs_):
            self.n_step_buffers[env_idx].append(
                (obs[env_idx], next_obs[env_idx], action[env_idx], reward[env_idx], done[env_idx]))

            if len(self.n_step_buffers[env_idx]) >= self.n_steps or done[env_idx]:
                discounted_reward = 0
                for i in range(len(self.n_step_buffers[env_idx])):
                    discounted_reward += self.n_step_buffers[env_idx][i][3] * (self.gamma ** i)
                    if self.n_step_buffers[env_idx][i][4]:  # If done, break early
                        break

                obs_, next_obs_, action_ = self.n_step_buffers[env_idx][0][:3]

                # Adding with max priority initially
                super(PrioritizedMultiStepReplayBuffer, self).add(
                    np.expand_dims(obs_, axis=0),
                    np.expand_dims(next_obs_, axis=0),
                    np.expand_dims(action_, axis=0),
                    np.expand_dims(discounted_reward, axis=0),
                    np.expand_dims(done[env_idx], axis=0),
                    []  # [infos[env_idx]]
                )
                idx = self.pos - 1
                self.sum_tree[idx] = self.max_priority ** self.alpha
                self.min_tree[idx] = self.max_priority ** self.alpha

                self.n_step_buffers[env_idx].pop(0)

            if done[env_idx]:
                self.n_step_buffers[env_idx] = []

    def sample(self, batch_size):
        indices = self._sample_proportional(batch_size)
        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size()) ** (-self.beta)

        for idx in indices:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()
            weight = (p_sample * self.size()) ** (-self.beta)
            weights.append(weight / max_weight)

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        obs, actions, rewards, next_obs, dones, infos = super(PrioritizedMultiStepReplayBuffer, self).sample(batch_size)
        return obs, actions, rewards, next_obs, dones, infos, indices, torch.tensor(weights, device=self.device)

    def _sample_proportional(self, batch_size):
        indices = []
        p_total = self.sum_tree.sum(0, self.size() - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority >= 0
            assert 0 <= idx < self.size()

            priority = priority ** self.alpha + self.epsilon
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

            self.max_priority = max(self.max_priority, priority)


class SimplifiedPrioritizedMultiStepReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device, optimize_memory_usage=False,
                 handle_timeout_termination=False, n_envs=1, n_steps=3, gamma=0.99, alpha=0.5, beta=0.5,
                 beta_increment_per_sampling=0):
        assert not optimize_memory_usage, "Memory optimization is not supported for multi-step replay buffers"
        super(SimplifiedPrioritizedMultiStepReplayBuffer, self).__init__(buffer_size, observation_space, action_space,
                                                                         device, 1,
                                                                         optimize_memory_usage,
                                                                         handle_timeout_termination)
        self.n_envs_ = n_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.epsilon = 1e-6
        self.n_step_buffers = [[] for _ in range(n_envs)]
        self.max_priority = 1.0
        self.beta_increment_per_sampling = beta_increment_per_sampling

        # Prioritized replay buffer structures
        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, obs, next_obs, action, reward, done, infos):
        for env_idx in range(self.n_envs_):
            self.n_step_buffers[env_idx].append(
                (obs[env_idx], next_obs[env_idx], action[env_idx], reward[env_idx], done[env_idx]))

            if len(self.n_step_buffers[env_idx]) >= self.n_steps or done[env_idx]:
                discounted_reward = 0
                for i in range(len(self.n_step_buffers[env_idx])):
                    discounted_reward += self.n_step_buffers[env_idx][i][3] * (self.gamma ** i)
                    if self.n_step_buffers[env_idx][i][4]:  # If done, break early
                        break

                obs_, next_obs_, action_ = self.n_step_buffers[env_idx][0][:3]

                # Add transition with maximum priority
                super(SimplifiedPrioritizedMultiStepReplayBuffer, self).add(
                    obs_,
                    next_obs_,
                    action_,
                    discounted_reward,
                    done[env_idx],
                    []
                )
                idx = (self.pos - 1) % self.buffer_size

                priority = self.max_priority ** self.alpha + self.epsilon
                self.sum_tree[idx] = priority
                self.min_tree[idx] = priority

                self.n_step_buffers[env_idx].pop(0)

            if done[env_idx]:
                self.n_step_buffers[env_idx] = []

    def sample(self, batch_size, beta=None):
        if beta is None:
            beta = self.beta
        assert self.size() > batch_size
        assert beta > 0

        indices = self._sample_proportional(batch_size)
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        weights /= weights.mean()

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        data = super(SimplifiedPrioritizedMultiStepReplayBuffer, self)._get_samples(indices)
        return data, indices, torch.tensor(weights, device=self.device, requires_grad=False)

    def update_priorities(self, indices, priorities):

        assert len(indices) == len(priorities)
        assert np.all(priorities >= 0)
        assert np.all((0 <= np.array(indices)) & np.all(np.array(indices) < self.size()))

        priorities = priorities ** self.alpha + self.epsilon

        for idx, priority in zip(indices, priorities):
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority

        self.max_priority = max(self.max_priority, np.max(priorities))

    def _sample_proportional(self, batch_size):
        indices = []
        p_total = self.sum_tree.sum(0, self.size() - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
        return indices

    def _calculate_weight(self, idx, beta):
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * self.size()) ** (-beta)

        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * self.size()) ** (-beta)
        weight = weight / max_weight

        return weight


#########

# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
                capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
            self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)
