"""JointBanditWorldModel — joint hypotheses for two-arm trap bandits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import WorldModel
from ...outcome import Outcome


OUTCOME_ZERO = 0
OUTCOME_ONE = 1
OUTCOME_CATASTROPHE = 2


@dataclass(frozen=True)
class JointBanditComponent:
    """One complete bandit world."""
    world_type: str  # "safe" or "risky"
    p1: float
    p2: float
    p_cat: float = 0.01


@dataclass
class JointBanditWorldModelParameters:
    """Mixture over complete joint bandit worlds."""
    components: list[JointBanditComponent]
    weights: np.ndarray


@dataclass
class JointBanditBeliefState:
    """Outcome counts by arm.

    history[action, outcome], where outcomes are zero, one, catastrophe.
    """
    history: np.ndarray


class JointBanditWorldModel(WorldModel):
    """World model for two-arm trap bandits with joint arm hypotheses.

    Unlike MultiBernoulliWorldModel, this keeps p1 and p2 in the same
    component so a risky world's trapped arm can depend on argmax(p1, p2).
    """

    def __init__(self, num_arms: int = 2, num_outcomes: int = 3):
        assert num_arms == 2, "trap-bandit experiments currently assume exactly two arms"
        assert num_outcomes == 3
        self.num_arms = num_arms
        self.num_outcomes = num_outcomes

    def make_params(
        self,
        components: list[JointBanditComponent],
        weights: np.ndarray | None = None,
    ) -> JointBanditWorldModelParameters:
        assert len(components) > 0
        for component in components:
            self._validate_component(component)
        if weights is None:
            weights = np.ones(len(components)) / len(components)
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == (len(components),)
        assert np.all(weights >= 0)
        assert weights.sum() > 0
        weights = weights / weights.sum()
        return JointBanditWorldModelParameters(list(components), weights)

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
        return np.array([
            self.component_outcome_probs(component, action) @ reward_function[action]
            for action in range(self.num_arms)
        ])

    def component_outcome_probs(
        self,
        component: JointBanditComponent,
        action: int,
    ) -> np.ndarray:
        p = component.p1 if action == 0 else component.p2
        if component.world_type == "risky" and action == self.trapped_arm(component):
            return np.array([1.0 - p - component.p_cat, p, component.p_cat])
        return np.array([1.0 - p, p, 0.0])

    def trapped_arm(self, component: JointBanditComponent) -> int:
        return 0 if component.p1 >= component.p2 else 1

    def to_str(self, params: JointBanditWorldModelParameters) -> str:
        return f"JointBandit({len(params.components)} components)"

    def _validate_component(self, component: JointBanditComponent) -> None:
        assert component.world_type in {"safe", "risky"}
        assert 0 <= component.p1 <= 1
        assert 0 <= component.p2 <= 1
        assert 0 <= component.p_cat <= 1
        assert component.p1 + component.p_cat <= 1
        assert component.p2 + component.p_cat <= 1
