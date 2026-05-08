"""Trap-bandit environment."""
from __future__ import annotations

from .base import BaseEnvironment
from ..infrabayesian.builders.trap_bandit import (
    OUTCOME_CATASTROPHE,
    OUTCOME_ONE,
    OUTCOME_ZERO,
)


class TrapBanditEnvironment(BaseEnvironment):
    """Two-arm bandit where the higher-bias arm may contain a catastrophe tail."""

    def __init__(
        self,
        *,
        p1: float,
        p2: float,
        risky: bool,
        p_cat: float = 0.01,
        catastrophe_reward: float = -1000.0,
        **kwargs,
    ):
        super().__init__(num_actions=2, **kwargs)
        self.p1 = p1
        self.p2 = p2
        self.risky = risky
        self.p_cat = p_cat
        self.catastrophe_reward = catastrophe_reward
        self.trapped_arm = 0 if p1 >= p2 else 1

    def _respond(self, probabilities, action: int) -> int:
        return self._sample_outcome(action)

    def _resolve(self, observation, action):
        if observation == OUTCOME_CATASTROPHE:
            return self.catastrophe_reward
        return float(observation == OUTCOME_ONE)

    def _sample_outcome(self, action: int) -> int:
        p = self.p1 if action == 0 else self.p2
        if self.risky and action == self.trapped_arm:
            u = self.random.random()
            if u < self.p_cat:
                return OUTCOME_CATASTROPHE
            if u < self.p_cat + p:
                return OUTCOME_ONE
            return OUTCOME_ZERO

        if self.random.random() < p:
            return OUTCOME_ONE
        return OUTCOME_ZERO

    def expected_value(self, action: int) -> float:
        p = self.p1 if action == 0 else self.p2
        if self.risky and action == self.trapped_arm:
            return p + self.p_cat * self.catastrophe_reward
        return p

    def get_optimal_reward(self) -> float:
        return max(self.expected_value(0), self.expected_value(1))
