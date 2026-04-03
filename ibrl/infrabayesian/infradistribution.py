"""Infradistribution — wraps AMeasure objects."""
import warnings

import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome
from .a_measure import AMeasure


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
        if 0.0 < g < 1.0:
            warnings.warn(
                f"g={g}: for 0 < g < 1, individual a-measure cohomogeneity (λ+b ≤ 1) "
                f"is not preserved by Definition 11 for non-worst-case measures. "
                f"Offsets compound exponentially, causing numerical blowup over many "
                f"steps. The infradistribution's lower envelope remains valid, but "
                f"results may degrade. Use g=1 (default) for stable behavior.",
                stacklevel=2,
            )
        self.measures = measures
        self.g = g

    # ── Public interface ──────────────────────────────────────────────

    def update(self, action: int, outcome: Outcome):
        """Definition 11 update: compute aggregates, then update each measure."""
        # Compute obs_prob for each measure
        outcome_probs = [m.belief.compute_outcome_probability(action, outcome)
                     for m in self.measures]
        # Compute normalization and worst case counterfactual
        worst_case_full = min(self._compute_full_value(m, op)
                              for m, op in zip(self.measures, outcome_probs))
        worst_case_cf = min(self._compute_counterfactual_value(m, op)
                            for m, op in zip(self.measures, outcome_probs))
        normalization = worst_case_full - worst_case_cf
        if normalization <= 0:
            raise ValueError(
                f"P^g_H(L) must be > 0 (observation has zero probability "
                f"under worst-case measure), got {normalization}")

        # Update each measure: scale & offset first (from pre-update values), then belief
        for m, obs_prob in zip(self.measures, outcome_probs):
            # (1) Offset update — absorb counterfactual surplus, normalize
            cf_surplus = self._compute_counterfactual_value(m, obs_prob) - worst_case_cf
            if cf_surplus < -1e-12:
                raise ValueError(
                    f"counterfactual_surplus must be ≥ 0, got {cf_surplus}")
            m.offset = max(0.0, cf_surplus) / normalization
            
            # (2) Scale update — rescale by P_k(obs), normalize
            m.scale = m.scale * obs_prob / normalization

            # (3) Enforce cohomogeneity: with g=1, λ+b=1 is an invariant
            # that exact arithmetic preserves but floating point doesn't.
            if float(self.g) == 1.0:
                assert abs(m.scale + m.offset - 1) < 1e-12
                m.scale = 1 - m.offset

            # (4) Bayesian update — belief conditions on observation
            m.belief.update(action, outcome)

    def evaluate(self) -> NDArray[np.float64]:
        # min across measures (axis=0), producing shape (num_actions,)
        models = [m.evaluate() for m in self.measures]
        return np.min(models, axis=0)
    
    # ── Private helpers ────────────────────────────────────────────────

    def _compute_counterfactual_value(self, m: AMeasure, obs_prob: float)-> float:
        """α_k((1-L) · g) = g · λ · P(not obs) + b"""
        return self.g * m.scale * (1.0 - obs_prob) + m.offset

    def _compute_full_value(self, m: AMeasure, obs_prob: float) -> float:
        """α_k(1 ★_L g) = λ · P(obs) + g · λ · P(not obs) + b"""
        return m.scale * obs_prob + self.g * m.scale * (1.0 - obs_prob) + m.offset

    def __repr__(self) -> str:
        return "[" + (",".join(str(m) for m in self.measures)) + "]"
