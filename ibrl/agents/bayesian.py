import numpy as np

from . import BaseGreedyAgent
from ..utils import dump_array


class BayesianAgent(BaseGreedyAgent):
    """
    Agent using Bayesian inference

    For each action, keep track of a normal distribution that corresponds to the uncertainty of the associated reward.
    Update this estimate at each iteration based on the observed information.
    Picks the action with the largest expected reward.
    """
    def get_probabilities(self) -> np.ndarray:
        return self.build_greedy_policy(self.values)

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        reward = outcome.reward
        # Estimate reward of the action and its uncertainty based on observed reward and priors
        # Define precision, tau, as 1/sigma^2 to avoid vanishing sigma precision issues
        self.values[action] = (self.precision[action] * self.values[action] + reward) / (self.precision[action] + 1.0)
        self.precision[action] += 1

    def reset(self):
        super().reset()
        self.values = np.zeros(self.num_actions)
        self.precision = np.ones(self.num_actions) * 0.1

    def dump_state(self):
        return dump_array(self.values)