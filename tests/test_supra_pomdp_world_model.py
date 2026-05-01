"""Tier 1 unit tests for SupraPOMDPWorldModel.

Tests cover each method in isolation using hand-crafted POMDPs whose
outputs can be computed analytically. Dependency order:
  param construction → initial state → update_state → posterior_weights
  → compute_likelihood → compute_expected_reward → compute_q_values
  → value_iteration → policy-dependent kernels.
"""
import numpy as np
import pytest

from ibrl.infrabayesian.world_models.supra_pomdp_world_model import (
    SupraPOMDPWorldModel, SupraPOMDPWorldModelBeliefState
)
from ibrl.outcome import Outcome


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _identity_pomdp(num_states=2, num_actions=1):
    """Fully-observable 2-state POMDP: identity T, identity B, uniform θ₀."""
    wm = SupraPOMDPWorldModel(num_states=num_states, num_actions=num_actions,
                               num_obs=num_states)
    T = np.eye(num_states)[np.newaxis, :, :].repeat(num_actions, axis=0)
    T = T.transpose(1, 0, 2)  # shape (|S|, |A|, |S|)
    B = np.eye(num_states)
    theta_0 = np.ones(num_states) / num_states
    R = np.zeros((num_states, num_actions, num_states))
    return wm, wm.make_params(T, B, theta_0, R)


def obs(o: int, reward: float = 0.) -> Outcome:
    return Outcome(reward=reward, observation=o)


# ── §3.1  Param construction and validation ───────────────────────────────────

def test_make_params_validates_shapes():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    B = np.eye(2)
    theta_0 = np.array([1.0, 0.0])
    R = np.zeros((2, 2, 2))
    wm.make_params(T, B, theta_0, R)  # must not raise


def test_make_params_rejects_nonstochastic_T():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    bad_T = np.ones((2, 2, 2))  # rows sum to 2
    with pytest.raises(AssertionError):
        wm.make_params(bad_T, np.eye(2), np.array([1., 0.]), np.zeros((2, 2, 2)))


def test_make_params_rejects_nonstochastic_B():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    bad_B = np.ones((2, 2))  # rows sum to 2
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    with pytest.raises(AssertionError):
        wm.make_params(T, bad_B, np.array([1., 0.]), np.zeros((2, 2, 2)))


def test_make_params_rejects_nonstochastic_theta_0():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    with pytest.raises(AssertionError):
        wm.make_params(T, np.eye(2), np.array([0.6, 0.6]),  # sum=1.2
                       np.zeros((2, 2, 2)))


def test_make_params_rejects_wrong_shape_R():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    with pytest.raises((AssertionError, ValueError)):
        wm.make_params(T, np.eye(2), np.array([1., 0.]),
                       np.zeros((2, 3, 2)))  # wrong shape


# ── §3.2  Initial state ───────────────────────────────────────────────────────

def test_initial_state_is_none():
    wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
    initial = wm.initial_state()
    assert initial.components is None


def test_is_initial_recognises_none():
    wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
    initial_state = wm.initial_state()
    assert wm.is_initial(initial_state)
    non_initial = SupraPOMDPWorldModelBeliefState([(np.array([1., 0., 0.]), 0.0)])
    assert not wm.is_initial(non_initial)


# ── §3.3  Belief filter — update_state ───────────────────────────────────────

def test_update_state_lazy_init_uses_theta_0():
    """First call from None uses θ₀ as the prior belief."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # Identity transitions; identity observation model
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # (|S|, |A|, |S|)
    B = np.eye(2)
    theta_0 = np.array([0.7, 0.3])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    state = wm.update_state(wm.initial_state(), obs(0), action=0, policy=None, params=params)
    belief, log_m = state.components[0]
    # obs=0, B=identity → only state 0 is consistent; posterior = [1, 0]
    np.testing.assert_allclose(belief, [1.0, 0.0])
    # marginal = θ₀[0] * B[0,0] + θ₀[1] * B[1,0] = 0.7*1 + 0.3*0 = 0.7
    assert np.isclose(log_m, np.log(0.7), atol=1e-9)


def test_update_state_pomdp_filter_textbook_example():
    """Tiger-problem-like: 2 states, noisy observation, identity transitions."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  # state-preserving
    B = np.array([[0.85, 0.15], [0.15, 0.85]])  # 85% accurate
    theta_0 = np.array([0.5, 0.5])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    # First observation: obs=0 (suggests state 0)
    state = wm.update_state(wm.initial_state(), obs(0), action=0, policy=None, params=params)
    belief, _ = state.components[0]
    np.testing.assert_allclose(belief, [0.85, 0.15])

    # Second observation: obs=0 again
    state = wm.update_state(state, obs(0), action=0, policy=None, params=params)
    belief, _ = state.components[0]
    expected = np.array([0.85 ** 2, 0.15 ** 2])
    expected /= expected.sum()
    np.testing.assert_allclose(belief, expected, atol=1e-9)


def test_update_state_handles_zero_likelihood_observation():
    """Observation with P=0 under a component: belief unchanged, log_m → -∞."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # obs=1 is impossible from state 0 (B[0,1]=0), and we start at state 0
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    B = np.array([[1.0, 0.0], [0.0, 1.0]])  # identity — obs=1 impossible from state 0
    theta_0 = np.array([1.0, 0.0])          # start deterministically in state 0
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    state = wm.update_state(wm.initial_state(), obs(1), action=0, policy=None, params=params)
    belief, log_m = state.components[0]
    assert not np.any(np.isnan(belief)), "belief must not contain NaN"
    assert log_m < -100, "log_marginal should be very negative for impossible obs"


def test_update_state_does_not_mutate_input_state():
    """update_state must be a pure function — original state unchanged."""
    wm, params = _identity_pomdp()
    state0 = wm.update_state(wm.initial_state(), obs(0), action=0, policy=None, params=params)
    belief_before = state0.components[0][0].copy()
    log_m_before = state0.components[0][1]

    _ = wm.update_state(state0, obs(1), action=0, policy=None, params=params)

    np.testing.assert_array_equal(state0.components[0][0], belief_before)
    assert state0.components[0][1] == log_m_before


# ── §3.4  Mixed params and posterior mixture weights  ─────────────────────────────────────

def _make_2state_params(wm, theta_0_vec):
    """Helper: identity T, identity B, specified θ₀."""
    T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    B = np.eye(2)
    theta_0 = np.array(theta_0_vec, dtype=float)
    R = np.zeros((2, 1, 2))
    return wm.make_params(T, B, theta_0, R)


def test_mix_params_preserves_components():
    """Two single-component params mixed with (0.6, 0.4) produce a 2-component mixture."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    p1 = _make_2state_params(wm, [1., 0.])
    p2 = _make_2state_params(wm, [0., 1.])
    mixed = wm.mix_params([p1, p2], np.array([0.6, 0.4]))
    assert len(mixed.weights) == 2
    assert np.isclose(mixed.weights[0], 0.6)
    assert np.isclose(mixed.weights[1], 0.4)


def test_mix_params_weights_sum_to_one():
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    p1 = _make_2state_params(wm, [1., 0.])
    p2 = _make_2state_params(wm, [0., 1.])
    mixed = wm.mix_params([p1, p2], np.array([0.6, 0.4]))
    assert np.isclose(mixed.weights.sum(), 1.0)


def test_mix_params_flattens_nested_mixtures():
    """Mixing a 2-component mixture with a 1-component mixture yields 3 components."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    p1 = _make_2state_params(wm, [1., 0.])
    p2 = _make_2state_params(wm, [0., 1.])
    p3 = _make_2state_params(wm, [0.5, 0.5])
    # inner mix: p1 (0.5) + p2 (0.5)
    m12 = wm.mix_params([p1, p2], np.array([0.5, 0.5]))
    # outer mix: m12 (0.7) + p3 (0.3) → 3 components with weights [0.35, 0.35, 0.30]
    mixed = wm.mix_params([m12, p3], np.array([0.7, 0.3]))
    assert len(mixed.weights) == 3
    np.testing.assert_allclose(sorted(mixed.weights), [0.30, 0.35, 0.35], atol=1e-9)


def test_posterior_weights_at_initial_state_equal_prior():
    """Before any observations, posterior weights equal prior weights."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    p1 = _make_2state_params(wm, [1., 0.])
    p2 = _make_2state_params(wm, [0., 1.])
    mixed_params = wm.mix_params([p1, p2], np.array([0.6, 0.4]))

    # Lazily initialise beliefs (call update_state from initial state)
    state = wm.update_state(wm.initial_state(), obs(0), action=0, policy=None,
                             params=mixed_params)
    # After one obs the weights are not equal to prior; test BEFORE first obs
    # by constructing the initial beliefs manually
    initial_beliefs = SupraPOMDPWorldModelBeliefState(
        [(wm._initial_belief(th0_k, None), 0.0) for th0_k in mixed_params.theta_0]
    )
    weights = wm._posterior_weights(initial_beliefs, mixed_params)
    np.testing.assert_allclose(weights, [0.6, 0.4], atol=1e-9)


def test_posterior_weights_shift_toward_truth_under_data():
    """Two competing components, one matches observations. Weight on truth → 1."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # Component 0: always in state 0 → obs=0 is certain
    # Component 1: always in state 1 → obs=1 is certain
    p0 = _make_2state_params(wm, [1., 0.])
    p1 = _make_2state_params(wm, [0., 1.])
    mixed_params = wm.mix_params([p0, p1], np.array([0.5, 0.5]))

    # Observe obs=0 repeatedly — should push weight to component 0
    state = wm.initial_state()
    for _ in range(30):
        state = wm.update_state(state, obs(0), action=0, policy=None,
                                 params=mixed_params)
    weights = wm._posterior_weights(state, mixed_params)
    assert weights[0] > 0.999, f"weight on truth should → 1, got {weights[0]}"
    assert weights[1] < 0.001


# ── §3.5  Likelihood ──────────────────────────────────────────────────────────

def test_compute_likelihood_at_initial_state():
    """At initial state, likelihood = Σ_{s,s'} θ₀[s] · T[s,a,s'] · B[s', o]."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    theta_0 = np.array([0.6, 0.4])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    # Hand compute: P(obs=0 | θ₀, a=0)
    # belief_pred[s'] = Σ_s θ₀[s]*T[s,0,s']
    belief_pred = theta_0 @ T[:, 0, :]       # [0.6*0.8+0.4*0.3, 0.6*0.2+0.4*0.7]
    expected_lik = float(belief_pred @ B[:, 0])
    got = wm.compute_likelihood(wm.initial_state(), obs(0), params, action=0, policy=None)
    assert np.isclose(got, expected_lik, atol=1e-9)


def test_compute_likelihood_marginalises_over_credal_mixture():
    """Two-component mixture: likelihood = Σ_k w_k · L_k."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T0 = np.array([[[1., 0.]], [[0., 1.]]])
    T1 = np.array([[[0., 1.]], [[1., 0.]]])
    B = np.eye(2)
    p0 = wm.make_params(T0, B, np.array([1., 0.]), np.zeros((2, 1, 2)))
    p1 = wm.make_params(T1, B, np.array([1., 0.]), np.zeros((2, 1, 2)))
    mixed = wm.mix_params([p0, p1], np.array([0.5, 0.5]))

    # p0: start s=0, T0 keeps s=0, obs=0 certain → L0=1.0
    # p1: start s=0, T1 moves to s=1, obs=0 has prob 0 → L1=0.0
    # mixture likelihood = 0.5*1.0 + 0.5*0.0 = 0.5
    got = wm.compute_likelihood(wm.initial_state(), obs(0), mixed, action=0, policy=None)
    assert np.isclose(got, 0.5, atol=1e-9)


def test_compute_likelihood_in_unit_interval():
    """For any valid params, likelihood ∈ [0, 1]."""
    rng = np.random.default_rng(0)
    wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=3)
    for _ in range(20):
        T = rng.dirichlet(np.ones(3), size=(3, 2))
        B = rng.dirichlet(np.ones(3), size=3)
        theta_0 = rng.dirichlet(np.ones(3))
        R = np.zeros((3, 2, 3))
        params = wm.make_params(T, B, theta_0, R)
        for o in range(3):
            for a in range(2):
                lik = wm.compute_likelihood(wm.initial_state(), obs(o), params, action=a,
                                            policy=None)
                assert 0.0 <= lik <= 1.0 + 1e-9


# ── §3.6  Value iteration ─────────────────────────────────────────────────────

def test_value_iteration_terminal_absorbing_state():
    """Two-state MDP: state 1 absorbs with R=1. V[1] = 1/(1-γ)."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, discount=0.9)
    # T: state 0 transitions to state 1; state 1 stays at state 1
    T = np.array([[[0., 1.]], [[0., 1.]]])  # (|S|, |A|, |S|)
    R = np.array([[[0., 1.]], [[0., 1.]]])  # R[s,a,s'] = 1 when s'=1
    policy = np.array([1.0])
    V = wm._value_iteration(T, R, policy)
    # From state 1: V[1] = R[1,0,1] + γ*V[1] → V[1] = 1/(1-0.9) = 10
    assert np.isclose(V[1], 10.0, atol=1e-4)
    # From state 0: V[0] = R[0,0,1] + γ*V[1] = 1 + 0.9*10 = 10
    assert np.isclose(V[0], 10.0, atol=1e-4)


def test_value_iteration_uniform_policy_known_solution():
    """2-state, 2-action MDP with known closed-form V under uniform π."""
    # State 0: action 0 → R=1, stay; action 1 → R=0, stay
    # State 1: action 0 → R=0, stay; action 1 → R=1, stay
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, discount=0.5)
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 1.; T[0, 1, 0] = 1.  # state 0 stays
    T[1, 0, 1] = 1.; T[1, 1, 1] = 1.  # state 1 stays
    R = np.zeros((2, 2, 2))
    R[0, 0, 0] = 1.  # s=0, a=0 → R=1
    R[1, 1, 1] = 1.  # s=1, a=1 → R=1
    policy = np.array([0.5, 0.5])  # uniform
    V = wm._value_iteration(T, R, policy)
    # Under uniform policy, both actions keep s=0 in s=0 (T[0,*,0]=1):
    # V[0] = 0.5*(1+γV[0]) + 0.5*(0+γV[0]) = 0.5 + γV[0] → V[0] = 0.5/(1-γ) = 1.0
    # Symmetric argument gives V[1] = 1.0.
    expected = 0.5 / (1 - 0.5)
    np.testing.assert_allclose(V, [expected, expected], atol=1e-6)


def test_value_iteration_converges_within_max_iter():
    """With γ < 1, value iteration always converges (no infinite loop)."""
    rng = np.random.default_rng(42)
    wm = SupraPOMDPWorldModel(num_states=4, num_actions=3, discount=0.95,
                               value_iter_max=5000)
    T = rng.dirichlet(np.ones(4), size=(4, 3))
    R = rng.random((4, 3, 4))
    policy = rng.dirichlet(np.ones(3))
    V = wm._value_iteration(T, R, policy)
    assert V.shape == (4,)
    assert not np.any(np.isnan(V))


def test_value_iteration_cache_hit():
    """Same (policy, params id) returns cached V without recomputing."""
    wm, params = _identity_pomdp()
    T_arr = params.T[0]
    R = params.R[0]
    policy = np.array([1.0])
    cache_key = (id(params), policy.tobytes())
    V1 = wm._value_iteration(T_arr, R, policy, cache_key=cache_key)
    # Mutate T_arr in place to confirm cached value is returned unchanged
    T_arr_original = T_arr.copy()
    # (We don't mutate in the test — just call again and verify same object)
    V2 = wm._value_iteration(T_arr, R, policy, cache_key=cache_key)
    assert V1 is V2, "cached result should be the exact same array object"


# ── §3.7  compute_expected_reward (one-step) ──────────────────────────────────

def test_compute_expected_reward_one_step_semantics():
    """compute_expected_reward returns E[rf[next_obs] | b, a], not multi-step value."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2,
                               discount=0.9)
    T = np.array([[[1., 0.]], [[0., 1.]]])  # identity
    B = np.eye(2)
    theta_0 = np.array([0.5, 0.5])
    R = np.ones((2, 1, 2))  # all rewards = 1 (irrelevant for this test)
    params = wm.make_params(T, B, theta_0, R)

    # rf = [1, 0]: reward for obs=0, no reward for obs=1
    # E[rf | belief=[1,0], a=0] = P(obs=0 | s=0) * 1 + P(obs=1 | s=0) * 0
    #   = B[0,0]*1 + B[0,1]*0 = 1.0
    state = wm.update_state(wm.initial_state(), obs(0), action=0, policy=None, params=params)
    # state belief is now [1, 0]
    result = wm.compute_expected_reward(state, np.array([1., 0.]), params,
                                        action=0, policy=None)
    assert np.isclose(result, 1.0, atol=1e-9)


def test_compute_expected_reward_zero_reward_function_gives_zero():
    """E_H(rf=0) == 0 regardless of belief or params."""
    wm, params = _identity_pomdp()
    rf_zero = np.zeros(2)
    result = wm.compute_expected_reward(wm.initial_state(), rf_zero, params,
                                        action=0, policy=None)
    assert np.isclose(result, 0.0, atol=1e-9)


def test_compute_expected_reward_equals_likelihood_when_rf_is_indicator():
    """E[1_{obs=o} | b, a] == P(obs=o | b, a) == compute_likelihood."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    theta_0 = np.array([0.6, 0.4])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T, B, theta_0, R)

    for o in range(2):
        rf_indicator = np.zeros(2)
        rf_indicator[o] = 1.0
        er = wm.compute_expected_reward(wm.initial_state(), rf_indicator, params,
                                        action=0, policy=None)
        lik = wm.compute_likelihood(wm.initial_state(), obs(o), params, action=0, policy=None)
        assert np.isclose(er, lik, atol=1e-9), (
            f"obs={o}: E[indicator] = {er} != likelihood = {lik}")


def test_compute_expected_reward_credal_average():
    """Mixed hypothesis: result is posterior-weighted sum of per-component E[rf]."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    T = np.array([[[1., 0.]], [[0., 1.]]])
    B = np.eye(2)
    p0 = wm.make_params(T, B, np.array([1., 0.]), np.zeros((2, 1, 2)))
    p1 = wm.make_params(T, B, np.array([0., 1.]), np.zeros((2, 1, 2)))
    mixed = wm.mix_params([p0, p1], np.array([0.5, 0.5]))
    rf = np.array([1., 0.])  # reward obs=0
    # p0 → belief=[1,0] → E[rf]=1.0; p1 → belief=[0,1] → E[rf]=0.0
    # mixture (equal weights) → E[rf] = 0.5
    result = wm.compute_expected_reward(wm.initial_state(), rf, mixed, action=0, policy=None)
    assert np.isclose(result, 0.5, atol=1e-9)


# ── §3.7b  compute_q_values (multi-step) ─────────────────────────────────────

def test_compute_q_values_fully_observable_recovers_state_value():
    """When B = identity, Q(b=[1,0], a) = Q_MDP(s=0, a) exactly.

    State 1 is absorbing with zero reward, so V[1]=0 and Q(s=0, a=1)=0.
    """
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2, discount=0.9)
    T = np.zeros((2, 2, 2))
    T[0, 0, 0] = 1.   # s=0, a=0 → s=0 (self-loop, gets reward)
    T[0, 1, 1] = 1.   # s=0, a=1 → s=1 (absorbing, no reward)
    T[1, 0, 1] = 1.   # s=1 absorbing under a=0
    T[1, 1, 1] = 1.   # s=1 absorbing under a=1
    R = np.zeros((2, 2, 2))
    R[0, 0, 0] = 1.   # reward only for s=0, a=0 → s=0
    B = np.eye(2)
    theta_0 = np.array([1., 0.])
    params = wm.make_params(T, B, theta_0, R)
    policy = np.array([1., 0.])  # always action 0

    Q = wm.compute_q_values(wm.initial_state(), params, policy=policy)
    # Under π=[1,0]: V[0] = 1 + γ*V[0] → V[0]=10; V[1]=0 (absorbing, no reward)
    # Q(b=[1,0], a=0) = R[0,0,0] + γ*V[0] = 1 + 0.9*10 = 10
    assert np.isclose(Q[0], 10.0, atol=1e-3)
    # Q(b=[1,0], a=1) = R[0,1,1] + γ*V[1] = 0 + 0 = 0
    assert np.isclose(Q[1], 0.0, atol=1e-3)


def test_compute_q_values_at_initial_belief():
    """Q(b=θ₀, a) = Σ_s θ₀[s] · Q_MDP(s, a)."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2, discount=0.9)
    T = np.array([[[0.5, 0.5]], [[0.5, 0.5]]])  # uniform transitions
    B = np.eye(2)
    theta_0 = np.array([0.6, 0.4])
    R = np.zeros((2, 1, 2))
    R[:, 0, 0] = 1.  # R=1 whenever next state is 0
    params = wm.make_params(T, B, theta_0, R)
    policy = np.array([1.0])

    Q = wm.compute_q_values(wm.initial_state(), params, policy=policy)
    # V[s] = E[R|s,a=0] + γ*E[V|s,a=0].  Both states have the same dynamics:
    # E[R] = 0.5*1 + 0.5*0 = 0.5;  E[V] = 0.5*V[0] + 0.5*V[1].
    # By symmetry V[0]=V[1]=V, so V = 0.5 + 0.9*V → V = 0.5/(1-0.9) = 5.
    V_expected = 0.5 / (1 - 0.9)
    assert np.isclose(Q[0], V_expected, atol=1e-4)


def test_compute_q_values_credal_mixture_averages_by_posterior():
    """Two-component mixture: Q = weighted sum of per-component Q values."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2, discount=0.5)
    T = np.array([[[1., 0.]], [[0., 1.]]])
    B = np.eye(2)
    R_high = np.zeros((2, 1, 2))
    R_high[0, 0, 0] = 1.  # component 0: R=1 from state 0
    R_low = np.zeros((2, 1, 2))  # component 1: R=0 everywhere
    p_high = wm.make_params(T, B, np.array([1., 0.]), R_high)
    p_low = wm.make_params(T, B, np.array([1., 0.]), R_low)
    mixed = wm.mix_params([p_high, p_low], np.array([0.5, 0.5]))

    Q = wm.compute_q_values(wm.initial_state(), mixed, policy=np.array([1.0]))
    # Both components start at state 0 with equal weight (equal log-marginals at t=0)
    # p_high: V[0] = 1/(1-0.5) = 2; Q = 2
    # p_low: V[0] = 0; Q = 0
    # weighted: Q = 0.5*2 + 0.5*0 = 1.0
    assert np.isclose(Q[0], 1.0, atol=1e-4)


# ── §3.8  Policy-dependent kernels ────────────────────────────────────────────

def test_make_params_accepts_callable_theta_0():
    """θ₀ as callable π → array of shape (|S|,) is accepted."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    B = np.eye(2)
    theta_0_fn = lambda pi: np.array([pi[0], pi[1]])
    R = np.zeros((2, 2, 2))
    wm.make_params(T, B, theta_0_fn, R)  # must not raise


def test_make_params_accepts_callable_T():
    """T as callable π → array of shape (|S|, |A|, |S|) is accepted."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T_fn = lambda pi: np.array([[[pi[0], pi[1]], [0.5, 0.5]],
                                  [[0.5, 0.5], [pi[0], pi[1]]]])
    B = np.eye(2)
    theta_0 = np.array([0.5, 0.5])
    R = np.zeros((2, 2, 2))
    wm.make_params(T_fn, B, theta_0, R)  # must not raise


def test_make_params_rejects_callable_theta_0_with_wrong_output_shape():
    """Callable that returns wrong shape on uniform policy must fail at make_params."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    bad_theta_fn = lambda pi: np.array([0.5, 0.3, 0.2])  # shape (3,), not (2,)
    with pytest.raises(AssertionError):
        wm.make_params(T, np.eye(2), bad_theta_fn, np.zeros((2, 2, 2)))


def test_initial_belief_uses_callable_theta_0_with_policy():
    """For callable θ₀, _initial_belief resolves it against the given policy."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    B = np.eye(2)
    theta_0_fn = lambda pi: np.array([pi[0], 1. - pi[0]])
    R = np.zeros((2, 2, 2))
    params = wm.make_params(T, B, theta_0_fn, R)

    policy = np.array([0.8, 0.2])
    belief = wm._initial_belief(params.theta_0[0], policy=policy)
    np.testing.assert_allclose(belief, [0.8, 0.2])


def test_update_state_uses_callable_theta_0_on_lazy_init():
    """First update_state from None with callable θ₀ uses supplied policy."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    T = np.array([[[0.9, 0.1], [0.5, 0.5]],
                   [[0.5, 0.5], [0.1, 0.9]]])
    B = np.eye(2)
    theta_0_fn = lambda pi: np.array([pi[0], 1. - pi[0]])  # θ₀ = policy
    R = np.zeros((2, 2, 2))
    params = wm.make_params(T, B, theta_0_fn, R)

    policy = np.array([0.9, 0.1])
    # With B=identity and policy=[0.9,0.1], θ₀=[0.9,0.1]
    # obs=0: only state 0 consistent → belief=[1,0]
    state = wm.update_state(wm.initial_state(), obs(0), action=0, policy=policy, params=params)
    belief, _ = state.components[0]
    np.testing.assert_allclose(belief, [1., 0.], atol=1e-9)


def test_update_state_uses_callable_T_each_step():
    """Callable T(π) is evaluated with the current policy at each step."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # T depends on policy: high π[0] → stay in current state; low → swap
    def T_fn(pi):
        stay = pi[0]
        swap = 1. - pi[0]
        return np.array([[[stay, swap]], [[swap, stay]]])

    B = np.eye(2)
    theta_0 = np.array([1., 0.])  # start in state 0
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T_fn, B, theta_0, R)

    # policy π=[1,0]: T keeps state 0 → after a=0, still in s=0 → obs=0 certain
    policy_stay = np.array([1.0])
    state_stay = wm.update_state(wm.initial_state(), obs(0), action=0, policy=policy_stay, params=params)
    belief_stay, _ = state_stay.components[0]
    np.testing.assert_allclose(belief_stay, [1., 0.], atol=1e-9)

    # policy π=[0,1] (represented as scalar 0): T swaps → s=0 → s=1 → obs=0 impossible
    policy_swap = np.array([0.0])
    state_swap = wm.update_state(wm.initial_state(), obs(0), action=0, policy=policy_swap, params=params)
    _, log_m_swap = state_swap.components[0]
    assert log_m_swap < -100  # obs=0 was impossible


def test_compute_likelihood_uses_callable_T():
    """compute_likelihood with callable T resolves kernels against the policy."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
    # T(pi): if pi[0]=1.0, stay; if pi[0]=0.0, swap states
    def T_fn(pi):
        stay = pi[0]; swap = 1. - pi[0]
        return np.array([[[stay, swap]], [[swap, stay]]])

    B = np.eye(2)
    theta_0 = np.array([1., 0.])
    R = np.zeros((2, 1, 2))
    params = wm.make_params(T_fn, B, theta_0, R)

    # policy that stays in state 0 → P(obs=0) should be 1
    lik_stay = wm.compute_likelihood(wm.initial_state(), obs(0), params, action=0,
                                     policy=np.array([1.0]))
    assert np.isclose(lik_stay, 1.0, atol=1e-9)

    # policy that swaps to state 1 → P(obs=0) should be 0
    lik_swap = wm.compute_likelihood(wm.initial_state(), obs(0), params, action=0,
                                     policy=np.array([0.0]))
    assert lik_swap < 1e-9


def test_value_iteration_cache_invalidates_on_policy_change_with_callable_T():
    """Different policies with callable T must produce different V; cache must hit on repeat."""
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2, discount=0.9)

    def T_fn(pi):
        stay = pi[0]; swap = 1. - pi[0]
        return np.array([[[stay, swap]], [[swap, stay]]])

    B = np.eye(2)
    theta_0 = np.array([1., 0.])
    R = np.zeros((2, 1, 2))
    R[0, 0, 0] = 1.  # reward only in state 0
    params = wm.make_params(T_fn, B, theta_0, R)

    policy_stay = np.array([1.0])
    policy_swap = np.array([0.0])

    T_stay = T_fn(policy_stay)
    T_swap = T_fn(policy_swap)

    key_stay = (id(params), policy_stay.tobytes())
    key_swap = (id(params), policy_swap.tobytes())

    V_stay = wm._value_iteration(T_stay, R, policy_stay, cache_key=key_stay)
    V_swap = wm._value_iteration(T_swap, R, policy_swap, cache_key=key_swap)

    # staying in state 0 yields higher value for state 0
    assert V_stay[0] > V_swap[0], "staying policy should yield higher V[0]"

    # third call with policy_stay should hit cache
    V_stay_cached = wm._value_iteration(T_stay, R, policy_stay, cache_key=key_stay)
    assert V_stay_cached is V_stay, "cache must return the exact same object"


def test_transparent_newcomb_epsilon_one_step_likelihood():
    """Transparent Newcomb with ε: P(box=full | one-box policy) ≈ 1-ε."""
    eps = 0.05
    wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
    # States: 0=predicted_one_box, 1=predicted_two_box
    # Observations: 0=box_full, 1=box_empty
    # θ₀(π): predictor predicts with ε error
    # one-box policy: π[0]=1.0 → should be predicted as one-boxer with prob 1-ε
    def theta_0_fn(pi):
        p_one_box = (1 - eps) * pi[0] + eps * 0.5
        return np.array([p_one_box, 1. - p_one_box])

    T = np.zeros((2, 2, 2))
    T[0, :, 0] = 1.  # state is absorbed (single-step)
    T[1, :, 1] = 1.
    B = np.eye(2)   # obs=0 ↔ state=0 (one-box predicted → box full)
    R = np.zeros((2, 2, 2))
    params = wm.make_params(T, B, theta_0_fn, R)

    one_box_policy = np.array([1.0, 0.0])
    # P(obs=box_full | one-box policy) should be ≈ (1-ε)*1 + ε*0.5*1
    lik = wm.compute_likelihood(wm.initial_state(), obs(0), params, action=0,
                                policy=one_box_policy)
    expected = (1 - eps) * 1.0 + eps * 0.5
    assert np.isclose(lik, expected, atol=1e-6), f"got {lik}, expected {expected}"
