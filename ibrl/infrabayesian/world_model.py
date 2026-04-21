"""WorldModel — model-family abstraction for Infradistribution."""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from ..outcome import Outcome


class WorldModel(ABC):
    """
    Defines the belief state type, update logic, likelihood computation,
    and hypothesis param construction for a model family. Stateless —
    belief state is owned by Infradistribution.

    Belief state and params are opaque to callers — structure is defined
    by each subclass and documented there.
    """

    @abstractmethod
    def make_params(self, *args, **kwargs):
        """Construct hypothesis params for one a-measure."""
        pass

    @abstractmethod
    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """
        Combine params from multiple a-measures into one mixed a-measure.
        Called by Infradistribution.mix. Each element of params_list is the
        params of one input a-measure; coefficients are the mixing weights.
        """
        pass

    @abstractmethod
    def event_index(self, outcome: Outcome) -> int:
        """
        Extract the discrete event index from an outcome.
        Used by the gluing operator and belief state update.
        """
        pass

    @abstractmethod
    def initial_state(self):
        """Return the initial belief state (no observations)."""
        pass

    @abstractmethod
    def update_state(self, state, outcome: Outcome, action: int,
                     params=None, policy: np.ndarray | None = None):
        """
        Return new belief state after observing outcome under agent action.
        Does not mutate state.
        params: needed by policy-dependent models (ignored by Bernoulli).
        policy: needed by policy-dependent models (ignored by Bernoulli).
        """
        pass

    @abstractmethod
    def is_initial(self, state) -> bool:
        """True if state is the initial (no observations) state."""
        pass

    @abstractmethod
    def compute_likelihood(self, belief_state, outcome: Outcome, params,
                           action: int, policy: np.ndarray | None = None) -> float:
        """
        P(outcome | belief_state, params, action) under this hypothesis.
        Returns a scalar in [0, 1].
        """
        pass

    @abstractmethod
    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
                                params, action: int,
                                policy: np.ndarray | None = None) -> float:
        """
        E[reward_function(outcome) | belief_state, params, action].
        Returns a scalar.
        """
        pass


class MultiBernoulliWorldModel(WorldModel):
    """
    World model for a multi-arm Bernoulli bandit.

    Belief state: integer array of shape (num_arms, num_outcomes),
                  state[a, i] = number of times outcome i was observed on arm a.

    Hypothesis params: list of num_arms per-arm params,
                       params[a] = (log_probs, coefficients) where
                       log_probs has shape (num_components, num_outcomes) and
                       coefficients has shape (num_components,).

    Construct params via make_params(arm_hypotheses) where arm_hypotheses[a] is
    either a single probability array (shape num_outcomes) or a list of such arrays.
    """

    def __init__(self, num_arms: int, num_outcomes: int = 2):
        self.num_arms = num_arms
        self.num_outcomes = num_outcomes

    def make_params(self, arm_hypotheses: list):
        """
        arm_hypotheses[a]: a single np.ndarray of shape (num_outcomes,) for a
        point hypothesis, or a list of such arrays for a mixture prior.
        """
        assert len(arm_hypotheses) == self.num_arms
        result = []
        for hyps in arm_hypotheses:
            if isinstance(hyps, np.ndarray):
                hyps = [hyps]  # single point hypothesis → wrap in list
            probs = np.stack(hyps)  # (num_components, num_outcomes)
            assert np.allclose(probs.sum(axis=1), 1), "Hypothesis probabilities must sum to 1"
            log_probs = np.log(np.maximum(probs, 1e-300))
            coefficients = np.ones(len(hyps)) / len(hyps)
            result.append((log_probs, coefficients))
        return result

    def mix_params(self, params_list: list, coefficients: np.ndarray):
        """Mix per-arm params independently across arms."""
        mixed = []
        for arm in range(self.num_arms):
            arm_params = [p[arm] for p in params_list]
            log_probs = np.concatenate([p[0] for p in arm_params], axis=0)
            coefs = np.concatenate([p[1] * c for p, c in zip(arm_params, coefficients)])
            coefs /= coefs.sum()
            mixed.append((log_probs, coefs))
        return mixed

    def event_index(self, outcome: Outcome) -> int:
        return int(round(outcome.reward * (self.num_outcomes - 1)))

    def initial_state(self) -> np.ndarray:
        return np.zeros((self.num_arms, self.num_outcomes), dtype=np.int64)

    def update_state(self, state: np.ndarray, outcome: Outcome, action: int,
                     params=None, policy: np.ndarray | None = None) -> np.ndarray:
        new_state = state.copy()
        new_state[action, self.event_index(outcome)] += 1
        return new_state

    def is_initial(self, state: np.ndarray) -> bool:
        return (state == 0).all()

    def compute_likelihood(self, belief_state: np.ndarray, outcome: Outcome,
                           params, action: int,
                           policy: np.ndarray | None = None) -> float:
        probs = self._predictive(belief_state[action], params[action])
        return float(probs[self.event_index(outcome)])

    def compute_expected_reward(self, belief_state: np.ndarray,
                                reward_function: np.ndarray,
                                params, action: int,
                                policy: np.ndarray | None = None) -> float:
        probs = self._predictive(belief_state[action], params[action])
        return float(probs @ reward_function)

    def _predictive(self, arm_counts: np.ndarray, arm_params) -> np.ndarray:
        """Posterior predictive P(next outcome | arm history) for mixture of categoricals."""
        log_probs, coefficients = arm_params
        lp = np.expand_dims((log_probs * arm_counts).sum(axis=1), axis=1)
        lp -= lp.max()  # shift for numerical stability before exp
        probs = coefficients @ np.exp(lp + log_probs)
        return probs / probs.sum()
