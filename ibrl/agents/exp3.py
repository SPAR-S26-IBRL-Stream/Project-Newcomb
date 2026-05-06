import numpy as np
from numpy.typing import NDArray

from . import BaseAgent
from ..utils import dump_array


class EXP3Agent(BaseAgent):
    def __init__(self, *args,
            gamma : float = 0.1,
            max_reward : float = 1,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.max_reward = max_reward
        self.eta = gamma / self.num_actions

    def reset_belief(self):
        super().reset_belief()
        # We store weights in log-space for numerical stability
        # log(1.0) = 0
        self.log_weights = np.zeros(self.num_actions)
        self.probs = np.ones(self.num_actions) / self.num_actions

    def get_probabilities(self):
        return self.probs

    def update(self, probabilities : NDArray[np.float64], action : int, reward : float):
        super().update(probabilities, action, reward)

        # 1. Internal scaling ensures the math stays in the [0, 1] 'safe zone'
        scaled_reward = reward / self.max_reward
        
        # 2. Importance Sampling
        # This prevents the update from becoming too large if probs[a] is small
        estimated_reward = scaled_reward / self.probs[action]

        # 3. Update log-weights using the learning rate (eta)
        self.log_weights[action] += self.eta * estimated_reward

        # 4. Numerically stable softmax
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        self.probs = (1 - self.gamma) * (weights / np.sum(weights)) + (self.gamma / self.num_actions)
        self.probs /= self.probs.sum()

    def dump_state(self):
        return dump_array(self.log_weights)