"""Infradistribution — wraps AMeasure objects."""
from __future__ import annotations
import itertools
import numpy as np

from .a_measure import AMeasure


class Infradistribution:
    """
    An infradistribution, represented by it extremal minimal points
    I.e. we only keep track of the boundaries of the convex hull of minimal points
    """
    def __init__(self, measures : list[AMeasure]):
        assert isinstance(measures, list)
        assert len(measures) > 0
        self.measures = measures
        # Store number of occurrences of each outcome
        self.history = np.zeros(self.measures[0].num_outcomes, dtype=np.int64)  # number so occurrences

    @classmethod
    def mix(cls, components : list[Infradistribution], coefficients : np.ndarray):
        """
        Initialise a mixed infradistribution as the linear combination of several infradistributions.
        This method handles the case where some infradistributions might have multiple a-measures and
        where some of the a-measures might already be mixtures themselves.
        """
        # Make sure that infradistributions are unused
        # Mixing used infradistributions would be more complicated
        for component in components:
            assert component.history.sum() == 0
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
                assert measure.num_outcomes == components[0].measures[0].num_outcomes

        new_measures = []
        assert np.isclose(coefficients.sum(), 1)

        # Iterate over all possible combinations of exactly one a-measure from each infradistribution
        for measures in itertools.product(*(component.measures for component in components)):
            # Create mixed a-measure, by mixing all a-measures according to the mixing coefficients
            # Take care to handle the case where some of the a-measure are themselves mixtures
            mix_probabilities = np.exp(np.concatenate([measure.log_probabilities for measure in measures], axis=0))
            mix_coefficients = np.concatenate([measure.coefficients*coefficients[i] for i,measure in enumerate(measures)])
            mix_coefficients /= mix_coefficients.sum()  # for numerics
            new_measures.append(AMeasure.mixed(mix_probabilities,mix_coefficients))
        return cls(new_measures)

    @classmethod
    def mixKU(cls, components : list[Infradistribution]):
        """
        Initialise an infradistribution as having Knightian Uncertainty between several infradistributions.
        """
        # Make sure that infradistributions are unused
        # Mixing used infradistributions would be more complicated
        for component in components:
            assert component.history.sum() == 0
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
                assert measure.num_outcomes == components[0].measures[0].num_outcomes

        return cls(sum((component.measures for component in components), start=[]))


    def evaluate(self, reward_function : np.ndarray) -> float:
        """
        Pessimistic expected value of a given reward function
        Defined as minimal expected value over all minimal points
        """
        return min(measure.evaluate(self.history, reward_function) for measure in self.measures)

    def update(self, reward_function : np.ndarray, event : int) -> None:
        """
        Update all a-measures upon seeing a certain event using a given reward function
        This is Definition 11 from Basic Inframeasure Theory
        """
        # Expectation values (need to be computed before updating anything)
        glued0 = Infradistribution._glue(0, event, reward_function)
        glued1 = Infradistribution._glue(1, event, reward_function)
        expect0 = self.evaluate(glued0)
        expect1 = self.evaluate(glued1)
        probability = expect1 - expect0
        assert probability > 0  # If probability==0, the measure should be removed (to it, nothing matters anymore)
        expect_m = [measure.evaluate(self.history, glued0) for measure in self.measures]

        # Raw update 1: Update history
        # Update normalisation to the amount that we cut off
        # This is the m*L term of the IB update rule
        for measure in self.measures:
            measure.scale *= measure.compute_probabilities(self.history)[event]
        self.history[event] += 1

        # Raw update 2: Add off-history reward to offset
        # The measure evaluation includes the offset, so we do an assignment here, rather than an addition
        for i,measure in enumerate(self.measures):
            measure.offset = expect_m[i]

        # Renormalisation
        for measure in self.measures:
            measure.offset -= expect0
            measure /= probability

    def reset(self):
        """
        Reset internal state
        """
        self.history *= 0
        for measure in self.measures:
            measure.reset()

    def __repr__(self) -> str:
        return repr(self.measures)

    @staticmethod
    def _glue(value : float, event : int, reward_function : np.ndarray) -> np.ndarray:
        """
        Gluing operator: value *^event reward_function
        """
        reward_function = reward_function.copy()
        reward_function[event] = value
        return reward_function
