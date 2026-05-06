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
    """

    def __init__(self, *args, beliefs: list[BaseBelief],
                 g: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(beliefs, list) or len(beliefs) == 0:
            raise ValueError("beliefs must be a non-empty list of BaseBelief")
        self._belief_templates = beliefs
        self._g = g

    def reset_belief(self):
        super().reset_belief()
        measures = [AMeasure(b.copy()) for b in self._belief_templates]
        self.infradist = Infradistribution(measures, g=self._g)

    def update(self, probabilities: NDArray[np.float64], action: int, reward_or_outcome) -> None:
        """MODEL phase: update beliefs with observation.

        Accepts either a float `reward` (main-branch interface) or an
        `Outcome` object. Wraps a float into Outcome(reward=...) for the
        infradistribution update.

        Non-KU short-circuit: with a single belief, the IB update reduces
        to a plain Bayesian update on that belief (the (scale, offset)
        bookkeeping is trivial). Bypassing infradist.update for this case
        avoids catastrophic cancellation in the worst_case_full -
        worst_case_cf computation when likelihoods are very small (e.g.
        Gaussian density on heavy-tailed reward outliers).
        """
        super().update(probabilities, action, reward_or_outcome)
        if isinstance(reward_or_outcome, Outcome):
            outcome = reward_or_outcome
        else:
            outcome = Outcome(reward=float(reward_or_outcome))
        if len(self.infradist.measures) == 1:
            self.infradist.measures[0].belief.update(action, outcome)
        else:
            self.infradist.update(action, outcome)

    def get_probabilities(self) -> NDArray[np.float64]:
        """MODEL then PLAN: get reward structure, solve for policy."""
        reward_model = self.infradist.evaluate()

        if reward_model.ndim == 1:
            values = reward_model
        elif reward_model.ndim == 2:
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
        if self.verbose > 1:
            return str(self.infradist)
        model = self.infradist.evaluate()
        return dump_array(model) if model.ndim == 1 else dump_array(np.diag(model))
