import numpy as np
from numpy.typing import NDArray

from . import BaseAgent
from ..utils import dump_array


class UCB1Agent(BaseAgent):
    """
    UCB1 (Auer et al., 2002) for stochastic bandits.

    Pulls each arm once first, then chooses argmax over
        q[a] + c * sqrt( 2 * log(t) / counts[a] )

    where t is the total number of pulls so far and c is an exploration
    constant (default 1.0 — the textbook UCB1).
    """

    def __init__(self, *args, c: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = float(c)

    def get_probabilities(self) -> NDArray[np.float64]:
        probs = np.zeros(self.num_actions)
        # Initial round-robin: each arm at least once
        unexplored = np.flatnonzero(self.counts == 0)
        if unexplored.size > 0:
            probs[unexplored[0]] = 1.0
            return probs

        t = max(int(self.counts.sum()), 1)
        ucb = self.q + self.c * np.sqrt(2.0 * np.log(t) / self.counts)
        a = int(np.argmax(ucb))
        probs[a] = 1.0
        return probs

    def update(self, probabilities: NDArray[np.float64], action: int, reward: float):
        super().update(probabilities, action, reward)
        self.counts[action] += 1
        self.q[action] += (reward - self.q[action]) / self.counts[action]

    def reset_belief(self):
        super().reset_belief()
        self.q = np.zeros(self.num_actions)
        self.counts = np.zeros(self.num_actions)

    def dump_state(self) -> str:
        return dump_array(self.q)
