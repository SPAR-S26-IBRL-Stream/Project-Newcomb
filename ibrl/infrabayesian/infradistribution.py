"""Infradistribution — wraps AMeasure objects."""
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome
from .a_measure import AMeasure


@dataclass
class _MeasureSnapshot:
    """Pre-update snapshot of one a-measure's state.

    Captured BEFORE beliefs update, because the KU adjustment
    needs the prior probabilities, not the posterior ones.
    """
    obs_prob: float       # μ_k(L)     — P(observation) under this belief
    scale: float          # λ_k        — exp(log_scale)
    offset: float         # b_k        — current offset
    not_obs_prob: float = field(init=False)  # μ_k(1-L) — derived

    def __post_init__(self):
        if not (0.0 <= self.obs_prob <= 1.0):
            raise ValueError(
                f"obs_prob must be in [0, 1], got {self.obs_prob}")
        if self.scale <= 0:
            raise ValueError(
                f"scale (λ) must be > 0, got {self.scale}")
        if self.offset < -1e-12:
            raise ValueError(
                f"offset (b) must be ≥ 0, got {self.offset}")
        self.not_obs_prob = 1.0 - self.obs_prob


class Infradistribution:
    """Infradistribution over belief-based a-measures.

    Non-KU (1 measure): returns that measure's model.
    KU (N measures): returns element-wise min over all models.

    The update rule implements Definition 11 from Basic Inframeasure Theory.
    See experiments/alaro/docs/20260324_coinlearning.md §7.2-7.4.
    """

    def __init__(self, measures: list[AMeasure], g: float = 1.0):
        if len(measures) == 0:
            raise ValueError("Must provide at least one measure")
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"g must be in [0, 1], got {g}")
        self.measures = measures
        self.g = g

    # ── Public interface ──────────────────────────────────────────────

    def update(self, action: int, outcome: Outcome):
        # KU update (Definition 11, §7.2)
        snapshots = self._snapshot_measures(action, outcome)
        normalization = self._observation_probability(snapshots)
        self._apply_ku_update(snapshots, normalization, action, outcome)

    def evaluate(self) -> NDArray[np.float64]:
        models = [m.evaluate() for m in self.measures]
        return np.min(models, axis=0)

    # ── Private: Definition 11 steps ──────────────────────────────────

    def _snapshot_measures(self, action, outcome):
        """Capture each measure's state BEFORE updating beliefs."""
        snapshots = []
        for m in self.measures:
            snapshots.append(_MeasureSnapshot(
                obs_prob=m.belief.compute_outcome_probability(action, outcome),
                scale=np.exp(m.log_scale),
                offset=m.offset,
            ))
        return snapshots

    def _counterfactual_value(self, snap: _MeasureSnapshot) -> float:
        """α_k((1-L) · g) — How much value does a-measure k assign to the
        non-observed outcome, weighted by g?

        = c · λ_k · P_k(not obs) + b_k
        """
        return self.g * snap.scale * snap.not_obs_prob + snap.offset

    def _full_observation_value(self, snap: _MeasureSnapshot) -> float:
        """α_k(1 ★_L g) — "1 on observed branch, g on non-observed branch."

        = λ_k · P_k(obs) + c · λ_k · P_k(not obs) + b_k
        """
        return (snap.scale * snap.obs_prob
                + self.g * snap.scale * snap.not_obs_prob
                + snap.offset)

    def _observation_probability(self, snapshots: list[_MeasureSnapshot]) -> float:
        """P^g_H(L) — The infradistribution's "probability" of the observation.

        = E_H(1 ★_L g) - E_H(0 ★_L g)
        = min_k[full_value_k] - min_k[counterfactual_value_k]
        """
        worst_case_full = min(self._full_observation_value(s) for s in snapshots)
        worst_case_counterfactual = min(self._counterfactual_value(s) for s in snapshots)
        prob = worst_case_full - worst_case_counterfactual
        if prob <= 0:
            raise ValueError(
                f"P^g_H(L) must be > 0 (observation has zero probability "
                f"under worst-case measure), got {prob}")
        return prob

    def _apply_ku_update(self, snapshots, normalization, action, outcome):
        """Apply Definition 11 to each a-measure.

        For each measure k:
          1. Bayesian update of belief (μ_k conditions on observation)
          2. Scale update: λ_new = λ · P_k(obs) / P^g_H(L)
          3. Offset update: absorb counterfactual value surplus, normalize
        """
        worst_case_counterfactual = min(
            self._counterfactual_value(s) for s in snapshots
        )

        for snap, m in zip(snapshots, self.measures):
            # (1) Bayesian update — belief conditions on observation
            m.belief.update(action, outcome)

            # (2) Scale update — rescale by P_k(obs), normalize
            m.log_scale = np.log(snap.scale * snap.obs_prob / normalization)

            # (3) Offset update — absorb counterfactual surplus, normalize
            counterfactual_surplus = (
                self._counterfactual_value(snap) - worst_case_counterfactual
            )
            if counterfactual_surplus < -1e-12:
                raise ValueError(
                    f"counterfactual_surplus must be ≥ 0, got "
                    f"{counterfactual_surplus}")
            m.offset = max(0.0, counterfactual_surplus) / normalization
