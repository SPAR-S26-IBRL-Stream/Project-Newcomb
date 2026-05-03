"""JointBanditWorldModel — joint hypotheses for finite-outcome bandits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import WorldModel
from ...outcome import Outcome


@dataclass(frozen=True)
class JointBanditComponent:
    """One complete finite-outcome bandit hypothesis.

    probs[action, outcome] is the probability of outcome after pulling action.
    metadata is optional experiment-specific information, ignored by the model.
    """
    probs: np.ndarray
    metadata: dict | None = None


@dataclass
class JointBanditWorldModelParameters:
    """Mixture over complete joint bandit hypotheses."""
    components: list[JointBanditComponent]
    weights: np.ndarray


@dataclass
class JointBanditBeliefState:
    """Outcome counts by action: history[action, outcome]."""
    history: np.ndarray


class JointBanditWorldModel(WorldModel):
    """World model for finite-outcome bandits with joint arm hypotheses.

    MultiBernoulliWorldModel factorizes uncertainty per arm. This model keeps
    each complete bandit as one component, so hypotheses can encode correlations
    or constraints across arms.
    """

    def __init__(self, num_arms: int, num_outcomes: int):
        assert num_arms >= 1
        assert num_outcomes >= 2
        self.num_arms = num_arms
        self.num_outcomes = num_outcomes

    def make_params(
        self,
        components: list[JointBanditComponent],
        weights: np.ndarray | None = None,
    ) -> JointBanditWorldModelParameters:
        assert len(components) > 0
        normalized_components = []
        for component in components:
            self._validate_component(component)
            normalized_components.append(
                JointBanditComponent(
                    np.asarray(component.probs, dtype=float),
                    metadata=component.metadata,
                )
            )
        if weights is None:
            weights = np.ones(len(components)) / len(components)
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == (len(components),)
        assert np.all(weights >= 0)
        assert weights.sum() > 0
        weights = weights / weights.sum()
        return JointBanditWorldModelParameters(normalized_components, weights)

    def mix_params(
        self,
        params_list: list[JointBanditWorldModelParameters],
        coefficients: np.ndarray,
    ) -> JointBanditWorldModelParameters:
        components: list[JointBanditComponent] = []
        weights = []
        for params, coeff in zip(params_list, coefficients):
            components.extend(params.components)
            weights.extend(params.weights * float(coeff))
        return self.make_params(components, np.asarray(weights, dtype=float))

    def event_index(self, outcome: Outcome, action: int) -> int:
        assert outcome.observation is not None
        event = int(outcome.observation)
        assert 0 <= event < self.num_outcomes
        return event

    def initial_state(self) -> JointBanditBeliefState:
        return JointBanditBeliefState(
            np.zeros((self.num_arms, self.num_outcomes), dtype=np.int64)
        )

    def update_state(
        self,
        state: JointBanditBeliefState,
        outcome: Outcome,
        action: int,
        policy: np.ndarray,
        params=None,
    ) -> JointBanditBeliefState:
        history = state.history.copy()
        history[action, self.event_index(outcome, action)] += 1
        return JointBanditBeliefState(history)

    def is_initial(self, state: JointBanditBeliefState) -> bool:
        return not state.history.any()

    def compute_likelihood(
        self,
        belief_state: JointBanditBeliefState,
        outcome: Outcome,
        params: JointBanditWorldModelParameters,
        action: int,
        policy: np.ndarray,
    ) -> float:
        probs = self.posterior_predictive(belief_state, params, action)
        return float(probs[self.event_index(outcome, action)])

    def compute_expected_reward(
        self,
        belief_state: JointBanditBeliefState,
        reward_function: np.ndarray,
        params: JointBanditWorldModelParameters,
        action: int,
        policy: np.ndarray,
    ) -> float:
        probs = self.posterior_predictive(belief_state, params, action)
        return float(probs @ reward_function)

    def posterior_component_weights(
        self,
        belief_state: JointBanditBeliefState,
        params: JointBanditWorldModelParameters,
    ) -> np.ndarray:
        log_w = np.log(np.maximum(params.weights, 1e-300))
        for k, component in enumerate(params.components):
            for action in range(self.num_arms):
                probs = self.component_outcome_probs(component, action)
                log_w[k] += float(
                    belief_state.history[action]
                    @ np.log(np.maximum(probs, 1e-300))
                )
        log_w -= log_w.max()
        weights = np.exp(log_w)
        return weights / weights.sum()

    def posterior_predictive(
        self,
        belief_state: JointBanditBeliefState,
        params: JointBanditWorldModelParameters,
        action: int,
    ) -> np.ndarray:
        weights = self.posterior_component_weights(belief_state, params)
        predictive = np.zeros(self.num_outcomes)
        for weight, component in zip(weights, params.components):
            predictive += weight * self.component_outcome_probs(component, action)
        return predictive / predictive.sum()

    def component_expected_rewards(
        self,
        component: JointBanditComponent,
        reward_function: np.ndarray,
    ) -> np.ndarray:
        if reward_function.ndim == 1:
            return component.probs @ reward_function
        return np.array([
            self.component_outcome_probs(component, action) @ reward_function[action]
            for action in range(self.num_arms)
        ])

    def component_outcome_probs(
        self,
        component: JointBanditComponent,
        action: int,
    ) -> np.ndarray:
        return component.probs[action]

    def to_str(self, params: JointBanditWorldModelParameters) -> str:
        return f"JointBandit({len(params.components)} components)"

    def _validate_component(self, component: JointBanditComponent) -> None:
        probs = np.asarray(component.probs, dtype=float)
        assert probs.shape == (self.num_arms, self.num_outcomes)
        assert np.all(probs >= 0)
        assert np.allclose(probs.sum(axis=1), 1)
