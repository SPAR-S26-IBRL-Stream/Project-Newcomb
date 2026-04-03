import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..utils import dump_array


class QLearningAgent(BaseGreedyAgent):
    """
    Classical Q-learning agent that interacts with a multi-armed bandit

    Arguments:
        learning_rate: Learning rate for Q-learning or None (or negative value) to use sample averages
    """
    def __init__(self, *args,
            learning_rate : float | None = 0.1,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.learning_rate = None if (isinstance(learning_rate,float) and learning_rate < 0) else learning_rate

    def get_probabilities(self) -> NDArray[np.float64]:
        return self.build_greedy_policy(self.q)

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        reward = outcome.reward
        if self.learning_rate is None:
            # sample average
            self.counts[action] += 1
            self.q[action] += (reward - self.q[action]) / self.counts[action]
        else:
            # Q-learning
            self.q[action] += self.learning_rate * (reward - self.q[action])

    def reset(self):
        super().reset()
        if self.learning_rate is None:
            self.counts = np.zeros((self.num_actions,))
        self.q = np.zeros((self.num_actions,))

    def dump_state(self):
        return dump_array(self.q)