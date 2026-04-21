"""Infradistribution — wraps AMeasure objects with a shared WorldModel."""
from __future__ import annotations
import itertools

import numpy as np

from .a_measure import AMeasure
from .world_model import WorldModel
from ..outcome import Outcome


class Infradistribution:
    """
    An infradistribution represented by its extremal minimal points (a-measures).
    E_H(f) = inf over a-measures of (λ·μ(f) + b).

    belief_state: sufficient statistic of observations so far; owned here,
                  structure defined by world_model.
    """

    def __init__(self, measures: list[AMeasure], world_model: WorldModel):
        assert isinstance(measures, list) and len(measures) > 0
        assert world_model is not None
        self.measures = measures
        self.world_model = world_model
        self.belief_state = self.world_model.initial_state()

    @classmethod
    def mix(cls, components: list[Infradistribution], coefficients: np.ndarray):
        """
        Classical (Bayesian) mixture over infradistributions.
        Takes the Cartesian product of a-measures across components, creating one
        mixed a-measure per combination. All components must be unused.
        """
        assert np.isclose(coefficients.sum(), 1)
        for component in components:
            assert component.world_model.is_initial(component.belief_state), \
                "Can only mix unused infradistributions"
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
        assert all(
            isinstance(c.world_model, type(components[0].world_model))
            for c in components
        ), "All components must share the same WorldModel type"

        new_measures = []
        for measures in itertools.product(*(c.measures for c in components)):
            mixed_params = components[0].world_model.mix_params(
                [m.params for m in measures], coefficients
            )
            new_measures.append(AMeasure(mixed_params))
        return cls(new_measures, world_model=components[0].world_model)

    @classmethod
    def mixKU(cls, components: list[Infradistribution]):
        """
        Knightian Uncertainty mixture: concatenates all a-measures from all
        components. E_H(f) = min over all a-measures (pessimistic). All
        components must be unused.
        """
        for component in components:
            assert component.world_model.is_initial(component.belief_state), \
                "Can only mix unused infradistributions"
            for measure in component.measures:
                assert np.isclose(measure.scale, 1)
                assert np.isclose(measure.offset, 0)
        assert all(
            isinstance(c.world_model, type(components[0].world_model))
            for c in components
        ), "All components must share the same WorldModel type"
        return cls(
            sum((component.measures for component in components), start=[]),
            world_model=components[0].world_model,
        )

    def evaluate_action(self, reward_function: np.ndarray,
                        action: int,
                        policy: np.ndarray | None = None) -> float:
        """
        Pessimistic expected reward: inf over a-measures of λ·E_μ[reward] + b.
        reward_function: 1D array of per-outcome rewards for this action.
        """
        return min(
            measure.evaluate_action(
                self.world_model, self.belief_state, reward_function,
                action=action, policy=policy,
            )
            for measure in self.measures
        )

    def update(self, reward_function: np.ndarray, outcome: Outcome,
               action: int, policy: np.ndarray | None = None) -> None:
        """
        Update all a-measures after observing outcome on the given action.
        Implements Definition 11 from Basic Inframeasure Theory.

        reward_function: 2D array of shape (num_actions, num_outcomes).
        action: the action taken this step (required — not optional).
        policy: the agent's mixed strategy; passed through to world_model
                (ignored by MultiBernoulliWorldModel, used by Newcomb).
        """
        event = self.world_model.event_index(outcome)
        rf = reward_function[action]
        glued0 = Infradistribution._glue(0, event, rf)
        glued1 = Infradistribution._glue(1, event, rf)

        expect0 = self.evaluate_action(glued0, action=action, policy=policy)
        expect1 = self.evaluate_action(glued1, action=action, policy=policy)
        prob = expect1 - expect0
        assert prob > 0, "Zero-probability event — this measure should be pruned"

        expect_m = [
            measure.evaluate_action(
                self.world_model, self.belief_state, glued0,
                action=action, policy=policy,
            )
            for measure in self.measures
        ]

        # Scale update: λ_k *= P(outcome | belief, params_k, action)
        for measure in self.measures:
            measure.scale *= self.world_model.compute_likelihood(
                self.belief_state, outcome, measure.params,
                action=action, policy=policy,
            )

        # Belief state update (before offset update, after scale update)
        self.belief_state = self.world_model.update_state(
            self.belief_state, outcome, action=action,
            params=self.measures[0].params, policy=policy,
        )

        # Offset update: b_k = E_k(glued0)
        for i, measure in enumerate(self.measures):
            measure.offset = expect_m[i]

        # Renormalisation
        for measure in self.measures:
            measure.offset -= expect0
            measure /= prob

    def reset(self):
        self.belief_state = self.world_model.initial_state()
        for measure in self.measures:
            measure.reset()

    def __repr__(self) -> str:
        return repr(self.measures)

    @staticmethod
    def _glue(value: float, event: int, reward_function: np.ndarray) -> np.ndarray:
        """Gluing operator: value *^event reward_function."""
        reward_function = reward_function.copy()
        reward_function[event] = value
        return reward_function
