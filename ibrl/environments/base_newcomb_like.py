import numpy as np
from numpy.typing import NDArray

from . import BaseEnvironment
from ..utils import sample_action


class BaseNewcombLikeEnvironment(BaseEnvironment):
    """
    Base class for Newcomb-like environments

    The predictor samples an action from the agent's probability distribution.
    For predicted action i and selected action j, the reward will be reward_table[i,j]

    Arguments:
        reward_table
    """
    def __init__(self, *args,
            reward_table : list[list[float]],
            **kwargs):
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2  # technical limitation for now
        assert self.num_actions == len(reward_table)
        self.reward_table = np.array(reward_table)

    def predict(self, probabilities : NDArray[np.float64]):
        prediction = sample_action(self.random, probabilities)
        self.rewards = self.reward_table[prediction,:]

    def interact(self, action : int) -> float:
        return self.rewards[action]

    def get_optimal_reward(self) -> float:
        # Compute the optimal reward, based on the full reward table
        # The reward is a quadratic function of the probability of taking action 0.
        # Thus, there are three policies that could potentially be optimal
        (a,b),(c,d) = self.reward_table.tolist()
        return max(
            a,  # always take action 0
            d,  # always take action 1
            (a*d-(b+c)**2/4)/(a+d-b-c) if (a+d-b-c) < 0 else float("-inf")
                # take action 0 with probability (b+c-2*d)/(b+c-a-d)/2
        )
