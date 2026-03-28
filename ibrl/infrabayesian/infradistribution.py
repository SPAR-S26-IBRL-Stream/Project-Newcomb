"""Infradistribution — wraps AMeasure objects."""
import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome
from .a_measure import AMeasure


class Infradistribution:
    """Infradistribution over belief-based a-measures.

    Non-KU (1 measure): returns that measure's model.
    KU (N measures): returns element-wise min over all models.
    """

    def __init__(self, measures: list[AMeasure]):
        self.measures = measures

    def update(self, action: int, outcome: Outcome, context: dict | None = None):
        for m in self.measures:
            m.update(action, outcome, context)

    def expected_reward_model(self, context: dict | None = None) -> NDArray[np.float64]:
        models = [m.expected_reward_model(context) for m in self.measures]
        return np.min(models, axis=0)
