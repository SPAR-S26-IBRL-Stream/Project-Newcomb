import numpy as np

from . import BaseGreedyAgent
from ..utils import dump_array


class DiscreteBayesianAgent(BaseGreedyAgent):
    """
    Agent using Bayesian inference over a discrete set of hypotheses
    This is intended for environments two discrete outcomes, with rewards 0 and 1.
    The structure could easily be extended to more outcomes and different rewards.

    Hypotheses are initialised uniformly in the interval [0,1] (reward probabilities)
    For each action, keep track of a probability distribution over these hypotheses.
    Update this distribution at each iteration based on the observed information.
    Picks the action with the largest expected reward.
    """
    def __init__(self, *,
            num_hypotheses : int = 5,
            **kwargs):
        super().__init__(**kwargs)
        self.num_hypotheses = num_hypotheses
        self.hypotheses = np.stack([                    # shape (num_hypotheses,num_outcomes=2)
            np.linspace(1., 0., self.num_hypotheses),   # probability of outcome 0
            np.linspace(0., 1., self.num_hypotheses)    # probability of outcome 1
        ],axis=-1)
        self.reward_function = np.array([0.,1.])        # reward per outcome

    def get_probabilities(self) -> np.ndarray:
        return self.build_greedy_policy(self._expected_rewards())

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        observation = int(outcome.reward > 0.5)  # discretise reward

        # Bayesian update of prior
        self.prior[action] *= self.hypotheses[:,observation]
        self.prior[action] /= self.prior[action].sum()

    def reset(self):
        super().reset()
        self.prior = np.ones((self.num_actions,self.num_hypotheses)) / self.num_hypotheses

    def dump_state(self):
        return dump_array(self.prior)

    def _expected_rewards(self) -> np.ndarray:
        return self.prior @ self.hypotheses @ self.reward_function
