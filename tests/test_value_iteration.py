"""Tests for value iteration and lower-expectation kernel."""
import numpy as np
import pytest

from ibrl.mdp.value_iteration import (
    value_iteration, lower_expectation_kernel,
    lower_expectation_value_iteration,
)


def _two_state_chain():
    """2-state, 2-action MDP: state 0 has actions L (stay) / R (go to terminal 1).
    Reward 0 in state 0, reward 1 at terminal state 1."""
    P = np.zeros((2, 2, 2))
    P[0, 0, 0] = 1.0  # action 0 (left) keeps you in state 0
    P[0, 1, 1] = 1.0  # action 1 (right) moves to state 1 (terminal)
    P[1, :, 1] = 1.0  # state 1 absorbs
    R = np.array([0.0, 1.0])
    terminal = np.array([0, 1], dtype=np.float64)
    return P, R, terminal


class TestStandardVI:
    def test_converges_on_two_state_chain(self):
        P, R, term = _two_state_chain()
        V, pi = value_iteration(P, R, gamma=0.95, terminal_mask=term)
        # V[0] = max(loop, jump-to-terminal). Jumping arrives at state 1 with
        # reward R[1] = 1.0 immediately (no discount on the arrival reward
        # because the simulator credits R[s'] on entering s'). So V[0] = 1.0.
        assert V[0] == pytest.approx(1.0, abs=1e-6)
        assert V[1] == pytest.approx(0.0)
        assert pi[0, 1] == 1.0

    def test_terminal_states_are_zero(self):
        P, R, term = _two_state_chain()
        V, _ = value_iteration(P, R, gamma=0.5, terminal_mask=term)
        assert V[1] == 0.0


class TestLowerExpectationKernel:
    def test_degenerate_polytope_matches_fixed_kernel(self):
        # When p_lower == p_upper, the LP returns that exact kernel.
        P = np.zeros((2, 2, 2))
        P[0, 0, 1] = 1.0
        P[0, 1, 0] = 0.5
        P[0, 1, 1] = 0.5
        P[1, :, 1] = 1.0
        V = np.array([0.0, 1.0])
        result = lower_expectation_kernel(P, P, V)
        np.testing.assert_allclose(result, P)

    def test_assigns_p_upper_to_lowest_value(self):
        # 3-state square polytope, 1 action. At state 0:
        # p_lower = [0.1, 0.1, 0.1], p_upper = [0.5, 0.5, 0.5].
        # value_for_sort = [0, 1, 2]. Worst case: max prob on cell 0
        # (lowest V), min on cell 2 (highest V).
        p_lo = np.tile(np.array([0.1, 0.1, 0.1]), (3, 1, 1))  # (3, 1, 3)
        p_hi = np.tile(np.array([0.5, 0.5, 0.5]), (3, 1, 1))
        V = np.array([0.0, 1.0, 2.0])
        result = lower_expectation_kernel(p_lo, p_hi, V)
        # After lower bounds: 0.3 used, 0.7 remaining. Fill from lowest V:
        # cell 0 slack 0.4 → take 0.4 → remaining 0.3.
        # cell 1 slack 0.4 → take 0.3 → remaining 0.
        # → [0.5, 0.4, 0.1].
        np.testing.assert_allclose(result[0, 0], [0.5, 0.4, 0.1], atol=1e-9)

    def test_infeasible_polytope_raises(self):
        # 3-state square polytope, 1 action. p_lower row sums to 1.5 > 1.
        p_lo = np.tile(np.array([0.5, 0.5, 0.5]), (3, 1, 1))
        p_hi = np.tile(np.array([0.6, 0.6, 0.6]), (3, 1, 1))
        with pytest.raises(ValueError, match="sum.*p_lower"):
            lower_expectation_kernel(p_lo, p_hi, np.array([0.0, 1.0, 2.0]))


class TestLowerExpectationVI:
    def test_degenerate_polytope_matches_standard_vi(self):
        P, R, term = _two_state_chain()
        V_std, _ = value_iteration(P, R, gamma=0.95, terminal_mask=term)
        V_robust, _ = lower_expectation_value_iteration(
            P, P, R, gamma=0.95, terminal_mask=term)
        np.testing.assert_allclose(V_robust, V_std, atol=1e-6)

    def test_robust_v_lower_than_standard_under_uncertainty(self):
        # Two-state chain with uncertainty in transition success
        P = np.zeros((2, 2, 2))
        P[0, 0, 0] = 1.0
        P[0, 1, 1] = 1.0  # action 1 succeeds with prob 1
        P[1, :, 1] = 1.0
        R = np.array([0.0, 1.0])
        term = np.array([0, 1], dtype=np.float64)

        # Polytope: action 1's success prob ∈ [0.5, 1.0]
        P_lo = P.copy()
        P_lo[0, 1, 1] = 0.5
        P_lo[0, 1, 0] = 0.5
        P_hi = P.copy()  # success at 1.0 (the standard kernel)
        V_std, _ = value_iteration(P, R, gamma=0.95, terminal_mask=term)
        V_robust, _ = lower_expectation_value_iteration(
            P_lo, P_hi, R, gamma=0.95, terminal_mask=term)
        # Worst case has lower V because the adversary picks success_prob = 0.5
        assert V_robust[0] < V_std[0]
