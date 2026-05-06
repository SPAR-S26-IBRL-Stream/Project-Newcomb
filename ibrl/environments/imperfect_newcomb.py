"""Imperfect Newcomb with a non-stationary predictor accuracy.

For Option B: tests whether IB-KU (multiple admissible reward-matrix beliefs)
beats single-belief Bayesian / Thompson when the predictor's accuracy α
shifts mid-run. The optimal action flips around α=0.5: at α > 0.5 one-box
(action 0) is optimal; at α < 0.5 two-box (action 1) is optimal. A Bayesian
agent that commits early to either side has to relearn when α flips; an IB
agent that keeps both regimes alive never commits.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from . import BaseEnvironment


class ImperfectNewcombEnvironment(BaseEnvironment):
    """Newcomb with parametric predictor accuracy α that flips mid-run.

    The predictor predicts the agent's chosen action correctly with prob
    `alpha_now`, else picks the other action. `alpha_now` is `alpha_high`
    for the first `flip_at` steps, then `alpha_low` thereafter. Both
    `alpha_high` and `alpha_low` straddle 0.5 so that the optimal action
    flips at the boundary.

    Reward matrix (predicted, action):
        V[0, 0] = boxB         = 10  (one-box, predicted right)
        V[0, 1] = boxB + boxA  = 15  (two-box, predicted wrong way; both)
        V[1, 0] = 0                 (one-box, predicted wrong)
        V[1, 1] = boxA         = 5  (two-box, predicted right)
    """

    def __init__(self, *args,
                 alpha_high: float = 0.95,
                 alpha_low: float = 0.20,
                 flip_at: int = 50,
                 boxA: float = 5.0,
                 boxB: float = 10.0,
                 **kwargs):
        # Override num_actions silently to 2 (Newcomb is 2-action)
        kwargs.setdefault("num_actions", 2)
        super().__init__(*args, **kwargs)
        assert self.num_actions == 2
        assert 0.0 <= alpha_low < 1.0 and 0.0 < alpha_high <= 1.0
        self.alpha_high = float(alpha_high)
        self.alpha_low = float(alpha_low)
        self.flip_at = int(flip_at)
        self.reward_table = np.array([[boxB, boxB + boxA],
                                      [0.0, boxA]], dtype=np.float64)
        # Internal step counter — drives the alpha schedule. Persists across
        # reset_episode so multi-episode runs see the flip.
        self._step_total = 0
        self._last_predicted = 0
        self._current_probs: NDArray[np.float64] | None = None

    def reset(self):
        super().reset()
        self._step_total = 0
        self._current_probs = None
        return None

    def reset_episode(self):
        # Per-episode: re-seed RNG only; do NOT reset _step_total so the
        # alpha schedule is shared across episodes.
        super().reset_episode()
        self._current_probs = None

    def predict(self, probabilities: NDArray[np.float64]) -> None:
        """Sample the predictor's prediction at the current α."""
        self._current_probs = probabilities
        # Pick the agent's likely action under their policy
        agent_action = int(np.argmax(probabilities)) if probabilities.max() > 0.5 \
            else int(self.random.choice(self.num_actions, p=probabilities))
        alpha_now = self._alpha_at_step()
        if self.random.random() < alpha_now:
            self._last_predicted = agent_action  # correct
        else:
            self._last_predicted = 1 - agent_action  # incorrect (2-action)

    def interact(self, action: int) -> float:
        reward = float(self.reward_table[self._last_predicted, action])
        self._step_total += 1
        return reward

    def _alpha_at_step(self) -> float:
        return self.alpha_high if self._step_total < self.flip_at else self.alpha_low

    def get_optimal_reward(self) -> float:
        """Expected reward under the optimal stationary policy at the
        *current* alpha. Used for regret. The optimal policy is pure under
        any α ≠ 0.5: one-box if α > 0.5, two-box otherwise."""
        a = self._alpha_at_step()
        ev_one_box = a * self.reward_table[0, 0] + (1 - a) * self.reward_table[1, 0]
        ev_two_box = a * self.reward_table[1, 1] + (1 - a) * self.reward_table[0, 1]
        return float(max(ev_one_box, ev_two_box))

    def get_predicted(self) -> int:
        """For agents that observe the predictor's prediction (e.g. via
        Outcome.env_action) — exposed so the simulator can pass it on."""
        return int(self._last_predicted)
