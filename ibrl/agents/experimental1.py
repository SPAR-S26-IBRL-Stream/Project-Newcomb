import numpy as np

from . import QLearningAgent
from ..utils import sample_action


class ExperimentalAgent1(QLearningAgent):
    """
    Instead of using the non-deterministic probability distribution from Q-learning, sample an action from it and
    return a deterministic distribution that chooses this action.
    Consequently, we only access the diagonal of the reward matrix.
    """
    def get_probabilities(self) -> np.ndarray:
        proto_probabilities = super().get_probabilities()
        action = sample_action(self.random, proto_probabilities)
        probabilities = np.zeros((self.num_actions,))
        probabilities[action] = 1
        return probabilities
