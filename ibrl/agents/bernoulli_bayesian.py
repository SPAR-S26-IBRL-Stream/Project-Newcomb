import numpy as np

from . import BaseGreedyAgent
from ..utils import dump_array


class BernoulliBayesianAgent(BaseGreedyAgent):
    """Agent using Bayesian inference with Beta-Bernoulli conjugate updates.

    For each action, maintains a Beta(alpha, beta) posterior over the
    arm's reward probability. Updates via conjugate rule on binary rewards.
    Picks the action with the largest posterior mean.
    """
    def get_probabilities(self) -> np.ndarray:
        return self.build_greedy_policy(self.values)

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        reward = outcome.reward
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward
        self.values[action] = self.alpha[action] / (self.alpha[action] + self.beta[action])

    def reset(self):
        super().reset()
        self.alpha = np.ones(self.num_actions)   # Beta(1,1) = uniform prior
        self.beta = np.ones(self.num_actions)
        self.values = self.alpha / (self.alpha + self.beta)

    def dump_state(self):
        return dump_array(self.values)
