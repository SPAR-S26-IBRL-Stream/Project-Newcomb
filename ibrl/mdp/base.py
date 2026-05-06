"""Abstract bases for the MDP layer."""
from abc import ABC, abstractmethod

import numpy as np


class MDPEnvironment(ABC):
    """Episodic, finite-state, finite-action MDP.

    State and action are integer indices in [0, num_states) and
    [0, num_actions). Episodes terminate when `step` returns done=True
    or the simulator's max-step budget is exhausted.

    `reset()` is a full reset — it samples or re-initialises any
    persistent environment state (e.g. the adversary's worst-case kernel).
    `reset_episode()` re-seeds the per-episode RNG only, preserving
    persistent state across episodes so multi-episode learning is
    well-defined. Subclasses override `reset_episode` only when their
    persistent state is meaningfully episode-conditioned (e.g. an
    adversary that re-picks against the agent's previous-episode
    visit distribution).
    """

    def __init__(self, num_states: int, num_actions: int, *,
                 seed: int = 0x89ABCDEF, verbose: int = 0):
        assert isinstance(num_states, int) and num_states >= 1
        assert isinstance(num_actions, int) and num_actions >= 1
        self.num_states = num_states
        self.num_actions = num_actions
        self.seed = seed
        self.verbose = verbose

    def reset(self) -> int:
        """Full reset: re-init persistent state and RNG. Returns initial state."""
        self.reset_episode()
        return self._initial_state()

    def reset_episode(self) -> int:
        """Episode boundary: re-seed RNG, preserve persistent state.

        Default: bump seed and re-init `self.random`. Returns initial state.
        Subclasses may override to update episode-conditioned state.
        """
        self.seed += 1
        self.random = np.random.default_rng(seed=self.seed)
        return self._initial_state()

    @abstractmethod
    def _initial_state(self) -> int:
        """Return the start state for a new episode."""

    @abstractmethod
    def step(self, action: int) -> tuple[int, float, bool]:
        """Execute one step.

        Returns:
            (next_state, reward, done) — done=True ends the episode.
        """

    @abstractmethod
    def get_oracle_value(self, gamma: float = 0.95) -> np.ndarray:
        """Oracle V*(s) for the env's true kernel. Used for regret computation
        by the simulator and (optionally) by adversaries that need V* to pick
        the worst case. Shape (num_states,)."""


class MDPAgent(ABC):
    """Episodic MDP agent with a separate plan/act/observe interface.

    Lifecycle per episode:
        agent.plan()                     # compute policy from current belief
        for each step:
            a = agent.act(state)
            ...env step...
            agent.observe(s, a, s', r, done)

    `reset()` is a full reset; `reset_belief()` wipes learned state only;
    `reset_episode()` re-seeds RNG and per-episode counters but preserves
    learned belief so multi-episode learning is well-defined.
    """

    def __init__(self, num_states: int, num_actions: int, *,
                 seed: int = 0x01234567, verbose: int = 0):
        assert isinstance(num_states, int) and num_states >= 1
        assert isinstance(num_actions, int) and num_actions >= 1
        self.num_states = num_states
        self.num_actions = num_actions
        self.seed = seed
        self.verbose = verbose

    def reset(self) -> None:
        self.reset_belief()
        self.reset_episode()

    def reset_belief(self) -> None:
        """Wipe learned belief. Default no-op — override if the agent has
        persistent learned state."""

    def reset_episode(self) -> None:
        """Re-seed RNG and zero per-episode counters."""
        self.step_count = 0
        self.seed += 1
        self.random = np.random.default_rng(seed=self.seed)

    @abstractmethod
    def plan(self) -> None:
        """Compute and cache an action policy from the current belief.
        Called once at the start of each episode."""

    @abstractmethod
    def act(self, state: int) -> int:
        """Select action at the given state. Must be deterministic given
        cached policy + RNG state."""

    @abstractmethod
    def observe(self, state: int, action: int, next_state: int,
                reward: float, done: bool) -> None:
        """Incorporate one transition. Default: subclasses with no online
        update may leave this as a no-op (see RobustDPAgent)."""
