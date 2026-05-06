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
        # Construction-time guard: if any belief returns a 2-D model
        # (i.e. a reward matrix indexed by [env_action, agent_action] —
        # NewcombLikeBelief), then `_solve_game` will be exercised by
        # `get_probabilities`. The 2-action closed-form maximin is the
        # only solver implemented; assert num_actions == 2 so a future
        # user with a 3-action 2-D belief gets a clear error here rather
        # than a silent wrong answer down the line.
        for b in beliefs:
            try:
                model = b.predict_rewards()
            except Exception:
                continue  # belief may need data before it can predict
            if hasattr(model, "ndim") and model.ndim == 2 and self.num_actions != 2:
                raise NotImplementedError(
                    f"InfraBayesianAgent._solve_game only implements the "
                    f"2-action closed-form maximin; got num_actions="
                    f"{self.num_actions} with a 2-D belief model. "
                    f"Restrict to num_actions=2 or extend _solve_game.")

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
        """MODEL then PLAN: get reward structure, solve for policy.

        For 1-D models (BernoulliBelief, GaussianBelief), the model is a
        per-arm value vector and we wrap it through `build_greedy_policy`
        as usual. For 2-D models (NewcombLikeBelief), the matrix is a
        Newcomb-like (env_action × agent_action) reward table; we solve
        the maximin game directly and return the maximin policy without
        adding ε-greedy exploration on top, because `build_greedy_policy`
        would distort the saddle point.
        """
        reward_model = self.infradist.evaluate()

        if reward_model.ndim == 1:
            return self.build_greedy_policy(reward_model)
        if reward_model.ndim == 2:
            return self._solve_game(reward_model)
        raise ValueError(f"Unexpected reward model shape: {reward_model.shape}")

    def _solve_game(self, V: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve a Newcomb-like 2×2 game and return the maximin policy.

        V is a (num_env_actions, num_agent_actions) reward matrix where
        V[i, j] is the agent's payoff if the predictor predicts i and the
        agent plays j. For 2-action games the maximin probability of
        action 0 is the closed form (see ExperimentalAgent2 for the same
        algebra at experimental2.py:46-54): pure strategies are p₀=1
        (yields V[0, 0]) or p₀=0 (yields V[1, 1]); a mixed-strategy
        saddle point exists when V[0,0] + V[1,1] − V[0,1] − V[1,0] < 0,
        with p₀* = (V[0,1] + V[1,0] − 2·V[1,1]) / (2·(V[0,1] + V[1,0] −
        V[0,0] − V[1,1])) and saddle value (V[0,0]·V[1,1] −
        ((V[0,1] + V[1,0]) / 2)²) / (V[0,0] + V[1,1] − V[0,1] − V[1,0]).
        Pick whichever of {pure-0, pure-1, mixed} has the highest
        guaranteed value.

        For ≥3 actions, raises NotImplementedError. The construction-
        time assertion in __init__ also guards this path.
        """
        if V.shape != (2, 2):
            raise NotImplementedError(
                f"_solve_game closed-form covers 2×2 only; got shape {V.shape}. "
                f"Add a maximin LP solver to extend.")
        a = float(V[0, 0]); b = float(V[0, 1])
        c = float(V[1, 0]); d = float(V[1, 1])
        # Candidate strategies: (p_action_0, guaranteed_value)
        candidates = [(1.0, a), (0.0, d)]
        denom = a + d - b - c
        if denom < 0 and (b + c - a - d) != 0:
            p0_mixed = (b + c - 2.0 * d) / (b + c - a - d) / 2.0
            if 0.0 < p0_mixed < 1.0:
                value_mixed = (a * d - ((b + c) / 2.0) ** 2) / denom
                candidates.append((p0_mixed, value_mixed))
        p0, _ = max(candidates, key=lambda s: s[1])
        return np.array([p0, 1.0 - p0], dtype=np.float64)

    def dump_state(self) -> str:
        if self.verbose > 1:
            return str(self.infradist)
        model = self.infradist.evaluate()
        return dump_array(model) if model.ndim == 1 else dump_array(np.diag(model))
