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


class BayesianUCB(ExplorationStrategy):
    """Posterior action-value quantile UCB for Bayesian component mixtures."""

    def __init__(self, quantile: float = 0.95):
        assert 0.0 <= quantile <= 1.0
        self.quantile = float(quantile)

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        component_values, weights = agent.posterior_component_action_values()
        scores = np.array([
            self._weighted_quantile(component_values[:, action], weights, self.quantile)
            for action in range(agent.num_actions)
        ])
        best = np.isclose(scores, scores.max())
        return best.astype(float) / best.sum()

    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        cumulative /= cumulative[-1]
        idx = int(np.searchsorted(cumulative, quantile, side="left"))
        return float(sorted_values[min(idx, len(sorted_values) - 1)])


class HypothesisThompsonSampling(ExplorationStrategy):
    """Sample one posterior component/hypothesis, then act greedily under it."""

    def get_probabilities(self, agent, values: np.ndarray) -> np.ndarray:
        component = agent.sample_component_from_posterior()
        sampled_values = agent.values_for_component(component)
        best = np.isclose(sampled_values, sampled_values.max())
        return best.astype(float) / best.sum()


# Backwards-compatible alias for the original export name.
ThompsonSampling = HypothesisThompsonSampling
