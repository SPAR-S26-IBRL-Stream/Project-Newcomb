"""InfraBayesianAgent — agent using infrabayesian inference."""
import numpy as np
from numpy.typing import NDArray

from . import BaseGreedyAgent
from ..infrabayesian.beliefs import BaseBelief
from ..infrabayesian.a_measure import AMeasure
from ..infrabayesian.infradistribution import Infradistribution
from ..outcome import Outcome
from ..utils import dump_array


class InfraBayesianAgent(BaseGreedyAgent):
    """Agent using infrabayesian inference.

    Initialized with beliefs (list of epistemic models), not an environment.
    A single-element list gives non-KU (standard Bayesian) behavior;
    multiple elements give Knightian uncertainty.

    get_probabilities() has two phases:
      1. MODEL: ask infradist to evaluate the reward function
      2. PLAN: solve for the best policy given that structure

    update() has one phase:
      1. MODEL: pass observation to infradist to update beliefs

    Only InfraBayesianAgent uses AMeasure/Infradistribution/Belief.
    Other agents are unaffected.
    """

    def __init__(self, *args, beliefs: list[BaseBelief],
                 g: float = 1.0, utility=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(beliefs, list) or len(beliefs) == 0:
            raise ValueError("beliefs must be a non-empty list of BaseBelief")
        self._belief_templates = beliefs
        self._g = g
        self._utility = utility

    def reset(self):
        super().reset()
        measures = [AMeasure(b.copy()) for b in self._belief_templates]
        self.infradist = Infradistribution(measures, g=self._g)

    def update(self, probabilities: NDArray[np.float64], action: int, outcome) -> None:
        """MODEL phase: update beliefs with observation."""
        context = {'step': self.step, 'policy': probabilities}
        super().update(probabilities, action, outcome)  # base agent sees raw reward

        if self._utility is not None:
            # Map reward to utility before passing to infradistribution
            mapped_reward = self._utility(outcome.reward)
            if not (0.0 <= mapped_reward <= 1.0):
                raise ValueError(
                    f"Utility mapping must produce values in [0, 1], "
                    f"got {mapped_reward} from reward {outcome.reward}")
            mapped_outcome = Outcome(
                reward=mapped_reward,
                env_action=outcome.env_action if hasattr(outcome, 'env_action') else None,
            )
            self.infradist.update(action, mapped_outcome, context)
        else:
            self.infradist.update(action, outcome, context)

    def get_probabilities(self) -> NDArray[np.float64]:
        """MODEL then PLAN: get reward structure, solve for policy."""
        context = {'step': self.step}

        # MODEL: evaluate the reward function under worst-case measure
        reward_model = self.infradist.evaluate(context)

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
        model = self.infradist.evaluate(context)
        return dump_array(model) if model.ndim == 1 else dump_array(np.diag(model))
