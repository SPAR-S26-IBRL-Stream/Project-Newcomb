import numpy as np
from numpy.typing import NDArray

from . import QLearningAgent
from ..utils import sample_action


class ExperimentalAgent3(QLearningAgent):
    """
    Solve NDP as MDP

    In an MDP, we try to find the optimal (discrete) action
    In an NDP, we try to find the optimal (continuous) probability distribution

    If we discretise the probability distributions, we can effectively turn an NDP into a (much larger) MDP

    Arguments:
        resolution: Number of intervals for discretisation
    """
    def __init__(self,
            num_actions : int,
            resolution : int = 6,
            *args, **kwargs):
        assert num_actions == 2  # technical limitation for now
        self.real_num_actions = num_actions
        self.resolution = int(resolution)
        num_actions = self.resolution+1
        super().__init__(num_actions, *args, **kwargs)

    def get_probabilities(self) -> NDArray[np.float64]:
        # The action taken by the underlying Q-learning agent corresponds to the probability of taking action 0
        self.proto_probabilities = super().get_probabilities()
        self.proto_action = sample_action(self.random, self.proto_probabilities)  # store for updating Q-learning agent
        probabilities = np.zeros((self.real_num_actions,))
        probabilities[0] = self.proto_action / self.resolution
        probabilities[1] = 1 - probabilities[0]
        return probabilities

    def update(self, probabilities : NDArray[np.float64], action : int, outcome):
        super().update(self.proto_probabilities, self.proto_action, outcome)


    def dump_state(self):
        return f"{super().dump_state()}, proto_action: {self.proto_action}"
