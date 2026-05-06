import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..outcome import Outcome
from ..utils import dump_array


class BernoulliBayesianAgent(BaseGreedyAgent):
    """Agent using Bayesian inference with Beta-Bernoulli conjugate updates.

    For each action, maintains a Beta(alpha, beta) posterior over the
    arm's reward probability. Updates via conjugate rule on binary rewards.
    Picks the action with the largest posterior mean.
    """
    def get_probabilities(self) -> NDArray[np.float64]:
        return self.build_greedy_policy(self.values)

    def update(self, probabilities, action, reward_or_outcome):
        super().update(probabilities, action, reward_or_outcome)
        reward = (reward_or_outcome.reward if isinstance(reward_or_outcome, Outcome)
                  else float(reward_or_outcome))
        self.alpha[action] += reward
        self.beta[action] += 1.0 - reward
        self.values[action] = self.alpha[action] / (self.alpha[action] + self.beta[action])

    def reset_belief(self):
        super().reset_belief()
        self.alpha = np.ones(self.num_actions)
        self.beta = np.ones(self.num_actions)
        self.values = self.alpha / (self.alpha + self.beta)

    def dump_state(self):
        return dump_array(self.values)
