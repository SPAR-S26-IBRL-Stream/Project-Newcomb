"""Tests for the simplified IUCB (MatrixUCB) agent.

Tests cover construction, cell statistics, confidence matrix computation,
minimax strategy, and integration tests on Damascus/Newcomb/Bandit environments.
"""
import numpy as np
import pytest

from ibrl.agents.matrix_ucb import MatrixUCBAgent, CellStats, compute_confidence_matrix, solve_minimax_strategy
from ibrl.outcome import Outcome
from ibrl.utils.construction import construct_agent, construct_environment
from ibrl.simulators.simulator import simulate


class TestMatrixUCBConstruction:
    """Test that MatrixUCBAgent can be constructed and initialized."""

    def test_construct_via_string(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("matrix-ucb", options)
        assert isinstance(agent, MatrixUCBAgent)

    def test_construct_with_params(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("matrix-ucb:confidence_scale=1.5", options)
        assert agent.confidence_scale == 1.5

    def test_reset_initializes_cells(self):
        """Reset should initialize K² cells with count 0."""
        agent = MatrixUCBAgent(num_actions=3, seed=42)
        agent.reset()
        assert len(agent.cells) == 9  # 3x3
        for (b, a), stats in agent.cells.items():
            assert stats.count == 0
            assert stats.sum == 0.0
        assert agent.total_rounds == 0

    def test_works_with_more_than_two_actions(self):
        """Unlike full IUCB, MatrixUCB should work with K > 2."""
        agent = MatrixUCBAgent(num_actions=4, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (4,)
        assert abs(probs.sum() - 1.0) < 1e-10


class TestCellStats:
    """Test CellStats tracking."""

    def test_initial_state(self):
        stats = CellStats()
        assert stats.count == 0
        assert stats.sum == 0.0
        assert stats.mean() == 0.0

    def test_update_single(self):
        stats = CellStats()
        stats.update(0.5)
        assert stats.count == 1
        assert stats.sum == 0.5
        assert stats.mean() == 0.5

    def test_update_multiple(self):
        stats = CellStats()
        stats.update(0.2)
        stats.update(0.8)
        assert stats.count == 2
        assert abs(stats.mean() - 0.5) < 1e-10


class TestUpdateIncrements:
    """Test that update() correctly updates the right cell."""

    def test_update_increments_correct_cell(self):
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        agent.reset()

        # Simulate: predictor plays 0, agent plays 1, reward 5.0
        probs = agent.get_probabilities()
        agent.update(probs, 1, Outcome(reward=5.0, env_action=0))

        assert agent.cells[(0, 1)].count == 1
        assert agent.total_rounds == 1
        # Other cells should be untouched
        assert agent.cells[(0, 0)].count == 0
        assert agent.cells[(1, 0)].count == 0
        assert agent.cells[(1, 1)].count == 0

    def test_update_without_env_action_uses_row_zero(self):
        """Standard bandits (env_action=None) should place obs in row 0."""
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        agent.reset()

        probs = agent.get_probabilities()
        agent.update(probs, 0, Outcome(reward=1.0, env_action=None))

        assert agent.cells[(0, 0)].count == 1
        assert agent.cells[(1, 0)].count == 0


class TestConfidenceMatrix:
    """Test the upper confidence bound matrix computation."""

    def test_unobserved_cells_have_ucb_one(self):
        """Unobserved cells should have UCB = 1.0 (maximally optimistic)."""
        cells = {}
        for b in range(2):
            for a in range(2):
                cells[(b, a)] = CellStats()

        P_upper = compute_confidence_matrix(cells, 2, 0)
        np.testing.assert_array_equal(P_upper, np.ones((2, 2)))

    def test_observed_cells_have_correct_ucb(self):
        """Observed cells should have mean + bonus, clipped to 1.0."""
        cells = {}
        for b in range(2):
            for a in range(2):
                cells[(b, a)] = CellStats()

        # Observe cell (0, 0) 100 times with mean reward 0.0 (scaled)
        for _ in range(100):
            cells[(0, 0)].update(0.0)

        P_upper = compute_confidence_matrix(cells, 2, 100, confidence_scale=2.0)

        # Cell (0, 0): mean=0.0, bonus=sqrt(2*ln(100)/100) ≈ 0.303
        expected_bonus = np.sqrt(2.0 * np.log(100) / 100)
        assert abs(P_upper[0, 0] - (0.0 + expected_bonus)) < 1e-6

        # Other cells: still 1.0 (unobserved)
        assert P_upper[0, 1] == 1.0
        assert P_upper[1, 0] == 1.0
        assert P_upper[1, 1] == 1.0

    def test_ucb_clipped_to_one(self):
        """UCB values should be clipped to 1.0."""
        cells = {}
        for b in range(2):
            for a in range(2):
                cells[(b, a)] = CellStats()

        # Observe cell (0, 0) once with high reward
        cells[(0, 0)].update(0.9)

        P_upper = compute_confidence_matrix(cells, 2, 1, confidence_scale=2.0)
        # mean=0.9, bonus=sqrt(2*ln(1)/1)=0 (ln(1)=0), so UCB=0.9
        # Actually ln(1) = 0, so bonus = 0
        assert P_upper[0, 0] == pytest.approx(0.9, abs=1e-6)


class TestMinimaxSolver:
    """Test the minimax strategy solver."""

    def test_symmetric_game(self):
        """Symmetric anti-coordination game should give p=0.5."""
        # Damascus-like: [[0, 10], [10, 0]] scaled to [-1, 1]
        P = np.array([[-1.0, 1.0], [1.0, -1.0]])
        value, x = solve_minimax_strategy(P)
        assert abs(value - 0.0) < 1e-6
        assert abs(x[0] - 0.5) < 1e-6
        assert abs(x[1] - 0.5) < 1e-6

    def test_dominant_strategy(self):
        """When one action dominates, should play it with probability 1."""
        # Action 1 is always better regardless of predictor action
        P = np.array([[0.0, 1.0], [0.0, 1.0]])
        value, x = solve_minimax_strategy(P)
        assert abs(value - 1.0) < 1e-6
        assert x[1] > 0.99

    def test_general_2x2(self):
        """Test a general 2x2 game."""
        P = np.array([[3.0, 1.0], [0.0, 2.0]])
        value, x = solve_minimax_strategy(P)
        # Minimax value: max_x min_b [x*3+(1-x)*1, x*0+(1-x)*2]
        # Row 0: 2x+1, Row 1: 2-2x. Equal at x=0.25, value=1.5
        assert abs(value - 1.5) < 1e-4
        assert abs(x[0] - 0.25) < 1e-4

    def test_3x3_game(self):
        """Minimax solver should work for K > 2."""
        # Rock-paper-scissors payoff
        P = np.array([
            [ 0.0, -1.0,  1.0],
            [ 1.0,  0.0, -1.0],
            [-1.0,  1.0,  0.0],
        ])
        value, x = solve_minimax_strategy(P)
        assert abs(value - 0.0) < 1e-4
        # Uniform strategy
        for i in range(3):
            assert abs(x[i] - 1.0/3) < 1e-4

    def test_returns_valid_distribution(self):
        """Strategy should be a valid probability distribution."""
        P = np.random.RandomState(42).randn(4, 4)
        value, x = solve_minimax_strategy(P)
        assert x.shape == (4,)
        assert np.all(x >= -1e-10)
        assert abs(x.sum() - 1.0) < 1e-6


class TestMinimaxFromUCBMatrix:
    """Test that the agent computes minimax strategy from the UCB matrix."""

    def test_initial_policy_is_valid(self):
        """With no observations, all cells are 1.0, any strategy is optimal."""
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (2,)
        assert np.all(probs >= -1e-10)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_policy_responds_to_observations(self):
        """After asymmetric observations, policy should shift."""
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        agent.reset()

        # Feed many observations making action 1 clearly better
        probs = agent.get_probabilities()
        for _ in range(50):
            # Predictor plays 0, agent plays 0 -> low reward
            agent.update(probs, 0, Outcome(reward=-0.5, env_action=0))
            # Predictor plays 0, agent plays 1 -> high reward
            agent.update(probs, 1, Outcome(reward=0.8, env_action=0))
            # Predictor plays 1, agent plays 0 -> low reward
            agent.update(probs, 0, Outcome(reward=-0.5, env_action=1))
            # Predictor plays 1, agent plays 1 -> high reward
            agent.update(probs, 1, Outcome(reward=0.8, env_action=1))

        probs = agent.get_probabilities()
        # Action 1 dominates in all rows, so agent should strongly prefer it
        assert probs[1] > 0.9, f"Expected action 1 to dominate, got probs={probs}"


class TestRewardScaling:
    """Test the _scale_reward method."""

    def test_no_reward_range_clips(self):
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        assert agent._scale_reward(0.5) == 0.5
        assert agent._scale_reward(2.0) == 1.0
        assert agent._scale_reward(-2.0) == -1.0

    def test_reward_range_scales(self):
        agent = MatrixUCBAgent(num_actions=2, seed=42, reward_range=(0.0, 10.0))
        # 0 -> -1, 10 -> 1, 5 -> 0
        assert abs(agent._scale_reward(0.0) - (-1.0)) < 1e-10
        assert abs(agent._scale_reward(10.0) - 1.0) < 1e-10
        assert abs(agent._scale_reward(5.0) - 0.0) < 1e-10

    def test_equal_range_returns_zero(self):
        agent = MatrixUCBAgent(num_actions=2, seed=42, reward_range=(5.0, 5.0))
        assert agent._scale_reward(5.0) == 0.0


class TestDumpState:
    """Test dump_state returns a string."""

    def test_dump_state_returns_string(self):
        agent = MatrixUCBAgent(num_actions=2, seed=42)
        agent.reset()
        state = agent.dump_state()
        assert isinstance(state, str)


@pytest.mark.slow
class TestMatrixUCBOnDamascus:
    """Integration test: MatrixUCB should learn to mix on Damascus.

    Damascus payoff: [[0, 10], [10, 0]]. Optimal is p=0.5, value=5.
    """

    def test_converges_to_mixing(self):
        """Agent should converge to p ≈ 0.5 on Damascus."""
        options = {
            "num_actions": 2, "num_steps": 500, "num_runs": 20,
            "seed": 42, "verbose": 0,
        }
        env = construct_environment("damascus", dict(options))
        agent = construct_agent("matrix-ucb:reward_range=0:10", dict(options))
        results = simulate(env, agent, options)

        # Check that the average probability of action 0 in the last 100 steps
        # is near 0.5 (mixed strategy)
        final_probs = results["probabilities"][:, -100:, 0]  # (runs, steps)
        mean_p0 = final_probs.mean()
        assert 0.3 < mean_p0 < 0.7, \
            f"Expected p≈0.5 on Damascus, got mean p(action 0)={mean_p0:.3f}"

    def test_achieves_positive_reward(self):
        """Agent should achieve reward > 1.0 on Damascus (optimal is 5)."""
        options = {
            "num_actions": 2, "num_steps": 500, "num_runs": 20,
            "seed": 42, "verbose": 0,
        }
        env = construct_environment("damascus", dict(options))
        agent = construct_agent("matrix-ucb:reward_range=0:10", dict(options))
        results = simulate(env, agent, options)

        avg_reward = results["average_reward"][0][-100:].mean()
        assert avg_reward > 1.0, \
            f"Expected reward > 1.0 on Damascus, got {avg_reward:.2f}"


@pytest.mark.slow
class TestMatrixUCBOnNewcomb:
    """Integration test: MatrixUCB on Newcomb's problem.

    Newcomb payoff: [[10, 15], [0, 5]]. Optimal value = 10 (one-box).
    """

    def test_achieves_good_reward(self):
        """Agent should achieve reward > 5.0 on Newcomb."""
        options = {
            "num_actions": 2, "num_steps": 500, "num_runs": 20,
            "seed": 42, "verbose": 0,
        }
        env = construct_environment("newcomb", dict(options))
        agent = construct_agent("matrix-ucb:reward_range=0:15", dict(options))
        results = simulate(env, agent, options)

        avg_reward = results["average_reward"][0][-100:].mean()
        assert avg_reward > 5.0, \
            f"Expected reward > 5.0 on Newcomb, got {avg_reward:.2f}"


@pytest.mark.slow
class TestMatrixUCBOnBandit:
    """Integration test: MatrixUCB on standard bandits (sublinear regret)."""

    def test_sublinear_regret(self):
        """MatrixUCB should achieve sublinear regret on standard bandits."""
        options = {
            "num_actions": 2, "num_steps": 1000, "num_runs": 20,
            "seed": 42, "verbose": 0,
        }
        env = construct_environment("bandit", dict(options))
        agent = construct_agent("matrix-ucb", dict(options))
        results = simulate(env, agent, options)

        cumulative = results["average_reward"][0].cumsum()
        optimal_cumulative = results["optimal_reward"] * np.arange(1, 1001)
        regret = optimal_cumulative - cumulative

        # Regret rate should decrease (sublinear)
        assert regret[999] / 1000 < 0.15, \
            f"Regret rate too high: {regret[999]/1000:.3f}"


@pytest.mark.slow
class TestMatrixUCBVsFullIUCB:
    """Comparison: MatrixUCB should outperform full IUCB at short horizons."""

    def test_better_than_iucb_at_short_horizon(self):
        """At N=200, MatrixUCB should get substantially better reward than IUCB on Damascus."""
        options = {
            "num_actions": 2, "num_steps": 200, "num_runs": 20,
            "seed": 42, "verbose": 0,
        }
        env = construct_environment("damascus", dict(options))

        # Run MatrixUCB
        agent_mucb = construct_agent("matrix-ucb:reward_range=0:10", dict(options))
        results_mucb = simulate(env, agent_mucb, options)
        reward_mucb = results_mucb["average_reward"][0][-50:].mean()

        # Run full IUCB
        agent_iucb = construct_agent("iucb:reward_range=0:10", dict(options))
        results_iucb = simulate(env, agent_iucb, options)
        reward_iucb = results_iucb["average_reward"][0][-50:].mean()

        assert reward_mucb > reward_iucb, \
            f"MatrixUCB ({reward_mucb:.2f}) should beat IUCB ({reward_iucb:.2f}) at N=200"
