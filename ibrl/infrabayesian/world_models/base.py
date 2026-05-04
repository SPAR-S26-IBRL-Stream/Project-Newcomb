"""WorldModel — model-family abstraction for Infradistribution."""
from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from ...outcome import Outcome


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
    def event_index(self, outcome: Outcome, action : int) -> int:
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
                     policy: np.ndarray, params=None):
        """
        Return new belief state after observing outcome under agent action.
        Does not mutate state.
        Policy needed by policy-dependent models (ignored by Bernoulli).
        params: hypothesis params for the a-measure being updated; required
            by stateful world models (e.g. SupraPOMDPWorldModel) that need
            POMDP kernel matrices to propagate the belief. Ignored by
            stateless world models (Bernoulli, Newcomb).
        """
        pass

    @abstractmethod
    def is_initial(self, state) -> bool:
        """True if state is the initial (no observations) state."""
        pass

    @abstractmethod
    def compute_likelihood(self, belief_state, outcome: Outcome, params,
                           action: int, policy: np.ndarray) -> float:
        """
        P(outcome | belief_state, params, action) under this hypothesis.
        Returns a scalar in [0, 1].
        """
        pass

    @abstractmethod
    def compute_expected_reward(self, belief_state, reward_function: np.ndarray,
                                params, action: int, policy: np.ndarray) -> float:
        """
        E[reward_function(outcome) | belief_state, params, action].
        Returns a scalar.
        """
        pass
