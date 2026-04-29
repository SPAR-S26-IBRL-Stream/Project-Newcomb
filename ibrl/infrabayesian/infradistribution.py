"""Infradistribution — wraps AMeasure objects."""
from __future__ import annotations
import itertools
import numpy as np

from .a_measure import AMeasure
from .world_model import WorldModel
from ..outcome import Outcome


class Infradistribution:
    """
    An infradistribution, represented by it extremal minimal points
    I.e. we only keep track of the boundaries of the convex hull of minimal points
    """
    def __init__(self, measures : list[AMeasure], world_model : WorldModel):
        assert isinstance(measures, list)
        assert len(measures) > 0
        self.measures = measures
        self.world_model = world_model
        self.belief_state = self.world_model.initial_state()

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
            assert component.world_model.is_initial(component.belief_state)
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
        assert all(isinstance(c.world_model, type(components[0].world_model)) for c in components)

        new_measures = []
        assert np.isclose(coefficients.sum(), 1)

        # Iterate over all possible combinations of exactly one a-measure from each infradistribution
        for measures in itertools.product(*(component.measures for component in components)):
            # Create mixed a-measure by delegating to world_model — handles mixtures internally
            mixed_params = components[0].world_model.mix_params(
                [m.params for m in measures], coefficients
            )
            new_measures.append(AMeasure(mixed_params))
        return cls(new_measures, components[0].world_model)

    @classmethod
    def mixKU(cls, components : list[Infradistribution]):
        """
        Initialise an infradistribution as having Knightian Uncertainty between several infradistributions.
        """
        # Make sure that infradistributions are unused
        # Mixing used infradistributions would be more complicated
        for component in components:
            assert component.world_model.is_initial(component.belief_state)
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
        assert all(isinstance(c.world_model, type(components[0].world_model)) for c in components)

        return cls(sum((component.measures for component in components), start=[]),
                   components[0].world_model)

    def evaluate_action(self, reward_function : np.ndarray, action : int,
                        policy : np.ndarray) -> float:
        """
        Pessimistic expected value of a given reward function
        Defined as minimal expected value over all minimal points
        """
        return min(measure.evaluate_action(self.world_model, self.belief_state,
                                           reward_function, action, policy)
                   for measure in self.measures)

    def update(self, reward_function : np.ndarray, outcome : Outcome,
               action : int, policy : np.ndarray) -> None:
        """
        Update all a-measures upon seeing a certain event using a given reward function
        This is Definition 11 from Basic Inframeasure Theory
        """
        event = self.world_model.event_index(outcome, action)
        rf = reward_function[action]

        # Expectation values (need to be computed before updating anything)
        glued0 = Infradistribution._glue(0, event, rf)
        glued1 = Infradistribution._glue(1, event, rf)
        expect0 = self.evaluate_action(glued0, action, policy)
        expect1 = self.evaluate_action(glued1, action, policy)
        probability = expect1 - expect0
        assert probability > 0  # If probability==0, the measure should be removed (to it, nothing matters anymore)
        expect_m = [measure.evaluate_action(self.world_model, self.belief_state, glued0,
                                            action, policy)
                    for measure in self.measures]

        # Raw update 1: Update belief state
        # Update normalisation to the amount that we cut off
        # This is the m*L term of the IB update rule
        for measure in self.measures:
            measure.scale *= self.world_model.compute_likelihood(
                self.belief_state, outcome, measure.params, action, policy)
        self.belief_state = self.world_model.update_state(
            self.belief_state, outcome, action, policy,
            params=self.measures[0].params)

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
        self.belief_state = self.world_model.initial_state()
        for measure in self.measures:
            measure.reset()

    def __repr__(self) -> str:
        return "[" + ", ".join(measure.to_str(self.world_model) for measure in self.measures) + "]"

    @staticmethod
    def _glue(value : float, event : int, reward_function : np.ndarray) -> np.ndarray:
        """
        Gluing operator: value *^event reward_function
        """
        reward_function = reward_function.copy()
        reward_function[event] = value
        return reward_function
