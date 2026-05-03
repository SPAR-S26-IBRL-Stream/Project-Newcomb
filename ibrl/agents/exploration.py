"""Composable exploration strategies for action-value agents."""
from __future__ import annotations

import numpy as np


class ExplorationStrategy:
    """Strategy interface for converting values/agent state into a policy."""

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Greedy(ExplorationStrategy):
    """Uniformly tie-broken greedy action selection."""

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        best = np.isclose(values, values.max())
        return best.astype(float) / best.sum()


class EpsilonGreedy(ExplorationStrategy):
    """Epsilon-greedy exploration with a fixed or scheduled epsilon."""

    def __init__(self, epsilon: float | tuple[float, float, float]):
        self.epsilon = epsilon

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        greedy = Greedy().get_probabilities(agent, values)
        eps = self._epsilon(agent)
        uniform = np.ones(agent.num_actions) / agent.num_actions
        return (1 - eps) * greedy + eps * uniform

    def _epsilon(self, agent) -> float:
        if isinstance(self.epsilon, tuple):
            start, rate, end = self.epsilon
            return max(start / (agent.step ** rate), end)
        return float(self.epsilon)


class UniformPrefixThen(ExplorationStrategy):
    """Uniform exploration for a fixed prefix, then delegate."""

    def __init__(self, prefix_steps: int, base_strategy: ExplorationStrategy):
        self.prefix_steps = prefix_steps
        self.base_strategy = base_strategy

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        if agent.step <= self.prefix_steps:
            return np.ones(agent.num_actions) / agent.num_actions
        return self.base_strategy.get_probabilities(agent, values)


class UCB(ExplorationStrategy):
    """Empirical UCB over realized scalar rewards."""

    def __init__(self, c: float = 2.0):
        self.c = c

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        unpulled = agent.action_counts == 0
        if unpulled.any():
            return unpulled.astype(float) / unpulled.sum()
        total = max(agent.step, 2)
        bonus = self.c * np.sqrt(np.log(total) / agent.action_counts)
        scores = agent.empirical_values + bonus
        best = np.isclose(scores, scores.max())
        return best.astype(float) / best.sum()


class ThompsonSampling(ExplorationStrategy):
    """Thompson sampling over a posterior mixture of complete components."""

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        component = agent.sample_component_from_posterior()
        sampled_values = agent.values_for_component(component)
        best = np.isclose(sampled_values, sampled_values.max())
        return best.astype(float) / best.sum()
