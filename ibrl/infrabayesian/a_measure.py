"""AMeasure — wraps a belief with (lambda, b) a-measure structure."""
import numpy as np
from numpy.typing import NDArray

from ..outcome import Outcome
from .beliefs import BaseBelief


class AMeasure:
    """Wraps a belief with the (lambda, b) structure needed for IB.

    In non-KU mode, lambda=1 and b=0, making this a pure pass-through.
    """

    def __init__(self, belief: BaseBelief, log_scale: float = 0.0, offset: float = 0.0):
        self.belief = belief
        self.log_scale = log_scale  # log(lambda)
        self.offset = offset        # b

    def update(self, action: int, outcome: Outcome):
        self.belief.update(action, outcome)

    def evaluate(self) -> NDArray[np.float64]:
        """α(f) = λ · μ(f) + b"""
        scale = np.exp(self.log_scale)
        return scale * self.belief.predict_rewards() + self.offset
