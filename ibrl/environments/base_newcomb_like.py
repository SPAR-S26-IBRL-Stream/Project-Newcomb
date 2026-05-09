import numpy as np

from . import BaseEnvironment
from ..utils import sample_action
from ..outcome import Outcome


class BaseNewcombLikeEnvironment(BaseEnvironment):
    """
    Base class for Newcomb-like environments

    The predictor samples an action from the agent's probability distribution.
    For predicted action i and selected action j, the reward will be reward_table[i,j]

    Arguments:
        reward_table with reward_table[i,j] = reward on prediction i and action j
        predictor_accuracy ∈ [0.5, 1] with 0.5 = random, 1.0 = perfect predictor
    """
    def __init__(self, *,
            num_actions : int = None,
            reward_table : list[list[float]],
            predictor_accuracy : float = 1.0,
            **kwargs):
        if num_actions is None:
            num_actions = len(reward_table)
        super().__init__(num_actions=num_actions, **kwargs)
        assert self.num_actions == 2  # technical limitation for now
        assert self.num_actions == len(reward_table)
        self.reward_table = np.array(reward_table)
        self.predictor_accuracy = float(predictor_accuracy)

    def step(self, probabilities : np.ndarray, action : int) -> Outcome:
        # Step 1: sample predicted action based on probabilities
        perfect_prediction = probabilities
        random_prediction = np.ones(self.num_actions) / self.num_actions
        prediction = perfect_prediction * (2*self.predictor_accuracy - 1) \
                    + random_prediction * (2 - 2*self.predictor_accuracy)
        predicted_action = sample_action(self.random, prediction)
        # Step 2: get reward based on predicted and selected action
        reward = self.reward_table[predicted_action, action]
        return Outcome(reward=reward, observation=predicted_action)

    def get_optimal_reward(self) -> float:
        # Compute the optimal reward, based on the full reward table
        # The reward is a quadratic function of the probability of taking action 0.
        # Thus, there are three policies that could potentially be optimal
        (a,b),(c,d) = self.reward_table.tolist()
        acc = self.predictor_accuracy

        # The reward is a quadratic polynomial in p
        # reward(p) = A + B*p + C*p^2 = D + E*(1-p) + F*(1-p)^2
        A = acc*(a - c) + c
        C = (2*acc - 1)*(a + d - b - c)
        D = acc*(d - b) + b
        B = D - A - C  # from p==1
        # Collect[({1 - p, p} (2*acc - 1) + {1/2, 1/2} (2 - 2*acc)) . {{a, b}, {c, d}} . {1 - p, p}, p, List, List] // FullSimplify

        # This polynomial is maximised either for p=0, p=1 or, if C<0, p=-B/2C
        return max(
            A,  # p=0, i.e. always take action 0
            D,  # p=1, i.e. always take action 1
            A - B**2/(4*C) if (C < 0 and 0 <= -B/(2*C) <= 1) else float("-inf")
                # take action 0 with finite probability
        )
