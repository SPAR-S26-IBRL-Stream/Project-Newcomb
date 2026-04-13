"""Infradistribution — wraps AMeasure objects."""
import numpy as np

from .a_measure import AMeasure


def glue(value : float, event : int, reward_function : np.ndarray) -> np.ndarray:
    """
    Gluing operator: value *^event reward_function
    """
    reward_function = reward_function.copy()
    reward_function[event] = value
    return reward_function


class Infradistribution:
    """
    An infradistribution, represented by it extremal minimal points
    I.e. we only keep track of the boundaries of the convex hull of minimal points
    """
    def __init__(self, measures : list[A_measure]):
        self.measures = measures
        # Store number of occurrences of each outcome
        self.history = np.zeros(self.measures[0].num_outcomes, dtype=np.int64)  # number so occurrences

    def expected_value(self, reward_function : np.ndarray) -> float:
        """
        Expected value over a given reward function
        Defined as expected value minimum over all minimal points
        """
        return min(measure.expected_value(self.history, reward_function) for measure in self.measures)

    def update(self, reward_function : np.ndarray, event : int) -> None:
        """
        Update all a-measures upon seeing a certain event using a given reward function
        This is Definition 11 from Basic Inframeasure Theory
        """
        # Expectation values (need to be computed before updating anything)
        glued0 = glue(0, event, reward_function)
        glued1 = glue(1, event, reward_function)
        expect0 = self.expected_value(glued0)
        expect1 = self.expected_value(glued1)
        prob = expect1 - expect0
        expect_m = [measure.expected_value(self.history, glued0) for measure in self.measures]

        # Raw update 1: Update history
        # Update normalisation to the amount that we cut off
        # This is the m*L term of the IB update rule
        for measure in self.measures:
            measure.scale *= measure.compute_probabilities(self.history)[event]
        self.history[event] += 1

        # Raw update 2: Add off-history reward to offset
        for i,measure in enumerate(self.measures):
            measure.offset += expect_m[i]

        # Renormalisation
        for measure in self.measures:
            measure.offset -= expect0
            measure /= prob

    def __repr__(self) -> str:
        return repr(self.measures)
