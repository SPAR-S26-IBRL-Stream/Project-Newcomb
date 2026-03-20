import numpy as np
from numpy.typing import NDArray

from . import BaseAgent
from ..utils import dump_array


class UCBAgent(BaseAgent):
    """Classic UCB1 agent.

    Selects arms by argmax of (empirical mean + confidence bound).
    Exploration is entirely driven by the confidence term — no epsilon-greedy
    or softmax. Returns a one-hot probability distribution from
    get_probabilities().

    Arguments:
        exploration: Confidence bound multiplier (default 2.0, matching
                     the thesis's analysis for rewards in [-1, +1]).
    """
    def __init__(self, *args,
            exploration : float = 2.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration = exploration

    def get_probabilities(self) -> NDArray[np.float64]:
        # Initialization: pull each arm once (round-robin)
        unpulled = np.where(self.counts == 0)[0]
        if len(unpulled) > 0:
            best = unpulled[0]
        else:
            # UCB selection: deterministic argmax
            t = self.counts.sum()
            ucb_values = self.q / self.counts + self.exploration * np.sqrt(np.log(t) / self.counts)
            # Break ties randomly
            max_val = ucb_values.max()
            best_actions = np.where(np.isclose(ucb_values, max_val))[0]
            best = self.random.choice(best_actions)

        probabilities = np.zeros(self.num_actions)
        probabilities[best] = 1.0
        return probabilities

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        self.counts[action] += 1
        self.q[action] += outcome.reward

    def reset(self):
        super().reset()
        self.q = np.zeros(self.num_actions)
        self.counts = np.zeros(self.num_actions)

    def dump_state(self):
        if self.counts.sum() == 0:
            return dump_array(self.q)
        means = np.where(self.counts > 0, self.q / self.counts, 0)
        return dump_array(means)
