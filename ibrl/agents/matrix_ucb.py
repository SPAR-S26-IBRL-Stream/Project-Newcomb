"""UCB for matrix games with unknown payoffs.

Implements Algorithm 1 from O'Donoghue, Lattimore, and Osband (2021),
"Matrix games with bandit feedback," UAI 2021.
https://arxiv.org/abs/2006.05145

This is also a simplification of Kosoy's IUCB algorithm (2024) to the
payoff matrix special case: per-cell Hoeffding intervals replace the
abstract slab-based confidence set, and a minimax LP replaces the
cycle-based optimistic theta computation.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog

from . import BaseAgent
from ..utils import dump_array


class CellStats:
    """Statistics for one cell (b, a) of the payoff matrix."""

    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, reward_scaled: float):
        self.count += 1
        self.sum += reward_scaled
        self.sum_sq += reward_scaled ** 2

    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count

    def variance(self) -> float:
        if self.count < 2:
            return 1.0
        mean = self.mean()
        return self.sum_sq / self.count - mean ** 2


def compute_confidence_matrix(
    cells: dict[tuple[int, int], CellStats],
    num_actions: int,
    total_rounds: int,
    confidence_scale: float = 2.0,
) -> np.ndarray:
    """Compute the optimistic (upper confidence bound) payoff matrix.

    For each cell, UCB = mean + sqrt(confidence_scale * ln(t) / n).
    Unobserved cells get UCB = 1.0 (maximum possible).
    All values clipped to [-1, 1].
    """
    P_upper = np.ones((num_actions, num_actions))

    if total_rounds < 1:
        return P_upper

    log_t = np.log(total_rounds)

    for b in range(num_actions):
        for a in range(num_actions):
            stats = cells.get((b, a))
            if stats is None or stats.count == 0:
                P_upper[b, a] = 1.0
            else:
                bonus = np.sqrt(confidence_scale * log_t / stats.count)
                P_upper[b, a] = min(stats.mean() + bonus, 1.0)

    return P_upper


def solve_minimax_strategy(P: np.ndarray) -> tuple[float, np.ndarray]:
    """Solve max_x min_b sum_a x[a] * P[b, a] via LP.

    The LP (maximizing v is equivalent to minimizing -v):
        min  -v
        s.t. sum_a x[a] * P[b, a] >= v   for all b
             sum_a x[a] = 1
             x[a] >= 0                    for all a

    Decision variables: [x[0], ..., x[K-1], v].
    """
    K = P.shape[0]

    # Objective: minimize -v
    c = np.zeros(K + 1)
    c[K] = -1.0

    # Inequality constraints: v - sum_a x[a]*P[b,a] <= 0 for each b
    A_ub = np.zeros((K, K + 1))
    A_ub[:, :K] = -P
    A_ub[:, K] = 1.0
    b_ub = np.zeros(K)

    # Equality constraint: sum_a x[a] = 1
    A_eq = np.zeros((1, K + 1))
    A_eq[0, :K] = 1.0
    b_eq = np.array([1.0])

    # Bounds: x[a] >= 0, v is unbounded
    bounds = [(0, None)] * K + [(None, None)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    x = result.x[:K]
    value = result.x[K]
    return value, x


class MatrixUCBAgent(BaseAgent):
    """UCB for matrix games with unknown payoffs.

    Implements Algorithm 1 from O'Donoghue, Lattimore, and Osband
    (2021), "Matrix games with bandit feedback," UAI 2021.
    https://arxiv.org/abs/2006.05145

    Each round, the agent constructs an upper confidence bound (UCB)
    payoff matrix from per-cell Hoeffding intervals, then plays the
    mixed minimax strategy of this optimistic matrix. Observes the
    opponent's action and reward, updates the corresponding cell.

    This is also a simplification of Kosoy's IUCB algorithm (2024,
    "Imprecise Multi-Armed Bandits," https://arxiv.org/abs/2405.05673)
    to the payoff matrix special case. Kosoy's general framework
    handles arbitrary linear imprecise bandits using cycles, abstract
    norms, and slab-based confidence sets in a 10D hypothesis space.
    For payoff matrix games, all of that machinery reduces to per-cell
    confidence intervals and a minimax solve — this algorithm.

    Differences from O'Donoghue et al. Algorithm 1:
    - Bonus term: we use sqrt(2 * ln(t) / n) (anytime UCB1) instead
      of sqrt(2 * log(2T²mk) / n) (fixed-horizon). Avoids requiring
      the time horizon T as input; costs at most a log factor in the
      regret bound.
    - Reward range: we scale to [-1, 1]; they use [0, 1]. Cosmetic.

    Regret bound (O'Donoghue et al. Theorem 1):
      WorstCaseRegret <= O~(sqrt(m * k * T))
    where m, k are action counts and T is the number of rounds.

    Arguments:
        num_actions:      K (same for agent and predictor)
        reward_range:     (min, max) for scaling rewards to [-1, 1]
        confidence_scale: multiplier for Hoeffding bonus (default 2.0)
    """

    def __init__(self, *args,
                 reward_range: tuple[float, float] | None = None,
                 confidence_scale: float = 2.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_range = reward_range
        self.confidence_scale = confidence_scale

    def reset(self):
        super().reset()
        self.cells = {}
        self.total_rounds = 0
        for b in range(self.num_actions):
            for a in range(self.num_actions):
                self.cells[(b, a)] = CellStats()

    def get_probabilities(self) -> NDArray[np.float64]:
        P_upper = compute_confidence_matrix(
            self.cells, self.num_actions,
            self.total_rounds, self.confidence_scale
        )
        _, x = solve_minimax_strategy(P_upper)
        return x

    def update(self, probabilities, action, outcome):
        super().update(probabilities, action, outcome)
        reward_scaled = self._scale_reward(outcome.reward)
        b = outcome.env_action if outcome.env_action is not None else 0
        self.cells[(b, action)].update(reward_scaled)
        self.total_rounds += 1

    def _scale_reward(self, reward: float) -> float:
        if self.reward_range is None:
            return np.clip(reward, -1.0, 1.0)
        r_min, r_max = self.reward_range
        if r_max <= r_min:
            return 0.0
        return float(np.clip(2 * (reward - r_min) / (r_max - r_min) - 1, -1, 1))

    def dump_state(self) -> str:
        if self.total_rounds == 0:
            return "MatrixUCB: no observations"
        means = np.zeros((self.num_actions, self.num_actions))
        counts = np.zeros((self.num_actions, self.num_actions))
        for (b, a), stats in self.cells.items():
            means[b, a] = stats.mean()
            counts[b, a] = stats.count
        return f"MatrixUCB t={self.total_rounds} means={dump_array(means.ravel())} counts={dump_array(counts.ravel())}"
