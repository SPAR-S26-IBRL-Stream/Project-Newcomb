"""InfraBayesianAgent — agent using infrabayesian inference."""
import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution
from ..utils import dump_array


class InfraBayesianAgent(BaseGreedyAgent):
    """
    Agent using infrabayesian inference
    This is intended for environments two discrete outcomes, with rewards 0 and 1.
    The structure could easily be extended to more outcomes and different rewards.

    For each action, the agent maintains an mixed infradistribution, which corresponds to a classical distribution
    over (non-mixed) infradistributions, where each infradistribution represents a hypothesis.

    Each non-mixed infradistribution can contain either a single a-measure, in which case it represents a concrete
    possible world, or multiple a-measures, in which case it represents Knightian Uncertainty between those worlds.

    By default, non-mixed infradistributions are initialised uniformly in the interval [0,1] (reward probabilities)
    Mixed infradistributions are initialised to uniform priors of the non-mixed ones without KU.
    This corresponds to the DiscreteBayesianAgent.
    """
    def __init__(self, *args,
            num_hypotheses : int = 5,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hypotheses = num_hypotheses
        self.hypotheses = np.stack([                    # shape (num_hypotheses,num_outcomes=2)
            np.linspace(1., 0., self.num_hypotheses),   # probability of outcome 0
            np.linspace(0., 1., self.num_hypotheses)    # probability of outcome 1
        ],axis=-1)
        self.reward_function = np.array([0.,1.])        # reward per outcome

    def reset(self):
        super().reset()
        self.dists = []
        for _ in range(self.num_actions):
            coefficients = np.ones(self.num_hypotheses) / self.num_hypotheses
            measure = AMeasure(self.hypotheses, coefficients)
            self.dists.append(Infradistribution([measure]))

    def update(self, probabilities: NDArray[np.float64], action: int, outcome) -> None:
        super().update(probabilities, action, outcome)
        assert outcome.outcome is not None

        self.dists[action].update(self.reward_function, outcome.outcome)

    def get_probabilities(self) -> NDArray[np.float64]:
        expected_rewards = np.array([dist.expected_value(self.reward_function) for dist in self.dists])
        return self.build_greedy_policy(expected_rewards)

    def dump_state(self) -> str:
        state = "["+",".join(dump_array(dist.history,"%d") for dist in self.dists)+"]"
        if self.verbose > 1:
            state += ";" + repr(self.dists)
        return state
