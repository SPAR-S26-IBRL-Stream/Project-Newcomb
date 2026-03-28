"""InfraBayesianAgent — agent using infrabayesian inference."""
import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..infrabayesian.beliefs import BaseBelief
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution
from ..utils import dump_array


class InfraBayesianAgent(BaseGreedyAgent):
    """Agent using infrabayesian inference.

    Initialized with a BELIEF (epistemic model), not an environment.
    Wraps the belief in AMeasure/Infradistribution.

    get_probabilities() has two phases:
      1. MODEL: ask infradist for the expected reward structure
      2. PLAN: solve for the best policy given that structure

    update() has one phase:
      1. MODEL: pass observation to infradist to update beliefs

    Only InfraBayesianAgent uses AMeasure/Infradistribution/Belief.
    Other agents are unaffected.
    """

    def __init__(self, *args, belief: BaseBelief, **kwargs):
        super().__init__(*args, **kwargs)
        self._belief_template = belief

    def reset(self):
        super().reset()
        belief = self._belief_template.copy()
        self.infradist = Infradistribution([
            AMeasure(belief)  # single measure, lambda=1, b=0
        ])

    def update(self, probabilities: NDArray[np.float64], action: int, outcome) -> None:
        """MODEL phase: update beliefs with observation."""
        context = {'step': self.step, 'policy': probabilities}
        super().update(probabilities, action, outcome)
        self.infradist.update(action, outcome, context)

    def get_probabilities(self) -> NDArray[np.float64]:
        """MODEL then PLAN: get reward structure, solve for policy."""
        context = {'step': self.step}

        # MODEL: get the reward structure from the infradistribution
        reward_model = self.infradist.expected_reward_model(context)

        # PLAN: convert reward structure into a policy
        if reward_model.ndim == 1:
            values = reward_model
        elif reward_model.ndim == 2:
            # Heuristic: use diagonal (reward when predictor correctly predicts).
            # TODO: proper game solving (find pi maximizing pi^T V pi)
            values = self._solve_game(reward_model)
        else:
            raise ValueError(f"Unexpected reward model shape: {reward_model.shape}")

        return self.build_greedy_policy(values)

    def _solve_game(self, V: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve Newcomb-like game: return per-arm values for greedy policy.

        Heuristic: return diagonal of V (expected reward when predictor
        correctly predicts action). Proper game solving can be added later.
        """
        return np.diag(V)

    def dump_state(self) -> str:
        context = {'step': self.step}
        model = self.infradist.expected_reward_model(context)
        return dump_array(model) if model.ndim == 1 else dump_array(np.diag(model))
