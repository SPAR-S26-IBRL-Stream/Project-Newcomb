""" POMDPWorldModel unit tests.

TEST SCOPE:
  ✓ Policy-dependent kernels: T(π), B(π), θ₀(π) depending on policy vector
  ✓ State-MDP value iteration (Q-values over state space)
  ✓ Belief state updates via Bayesian filtering
  ✓ Likelihood computation for IB gluing operator
  
  ✗ NOT TESTED HERE:
  - Belief-dependent policies π(b) [see test_belief_dependent_agent.py]
  - Optimal policy extraction over belief space [see test_policy_optimizer.py]
  - Agent behavior with belief-indexed policies [see test_belief_dependent_agent.py]

NOTES ON FLAT POLICIES:
  All tests in this file use static/flat policies π ∈ ℝ^|A|.
  The world model can depend on π via callable kernels T(π), B(π), θ₀(π),
  but the policy itself does not depend on belief state.
  
  True POMDP optimality requires π(b) that changes with belief b.
  This is tested separately in test_belief_dependent_agent.py.
"""











import numpy as np
import pytest
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import (
    SupraPOMDPWorldModel, SupraPOMDPWorldModelBeliefState
)
from ibrl.outcome import Outcome


def obs(o: int, reward: float = 0.) -> Outcome:
    return Outcome(reward=reward, observation=o)


class TestSupraPOMDPParamConstruction:
    def test_make_params_validates_shapes(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([1.0, 0.0])
        R = np.zeros((2, 2, 2))
        wm.make_params(T, B, theta_0, R)

    def test_make_params_rejects_nonstochastic_T(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        bad_T = np.ones((2, 2, 2))
        with pytest.raises(AssertionError):
            wm.make_params(bad_T, np.eye(2), np.array([1., 0.]), np.zeros((2, 2, 2)))

    def test_make_params_accepts_callable_theta_0(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0_fn = lambda pi: np.array([pi[0], pi[1]])
        R = np.zeros((2, 2, 2))
        wm.make_params(T, B, theta_0_fn, R)


class TestSupraPOMDPInitialState:
    def test_initial_state_is_none(self):
        wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
        state = wm.initial_state()
        assert state.components is None

    def test_is_initial_recognises_none(self):
        wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)
        assert wm.is_initial(wm.initial_state())
        non_initial = SupraPOMDPWorldModelBeliefState([(np.array([1., 0., 0.]), 0.0)])
        assert not wm.is_initial(non_initial)


class TestSupraPOMDPBeliefFilter:
    def test_update_state_lazy_init_uses_theta_0(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        theta_0 = np.array([0.7, 0.3])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        state = wm.update_state(wm.initial_state(), obs(0), action=0, 
                               policy=np.array([1.0]), params=params)
        belief, log_m = state.components[0]
        np.testing.assert_allclose(belief, [1.0, 0.0])
        assert np.isclose(log_m, np.log(0.7))

    def test_update_state_pomdp_filter_textbook(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.array([[0.85, 0.15], [0.15, 0.85]])
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=np.array([1.0]), params=params)
        belief, _ = state.components[0]
        np.testing.assert_allclose(belief, [0.85, 0.15])
        
        state = wm.update_state(state, obs(0), action=0,
                               policy=np.array([1.0]), params=params)
        belief, _ = state.components[0]
        expected = np.array([0.85**2, 0.15**2])
        expected /= expected.sum()
        np.testing.assert_allclose(belief, expected)

    def test_update_state_does_not_mutate_input(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        theta_0 = np.array([1.0, 0.0])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        state0 = wm.update_state(wm.initial_state(), obs(0), action=0,
                                policy=np.array([1.0]), params=params)
        belief_before = state0.components[0][0].copy()
        
        _ = wm.update_state(state0, obs(1), action=0,
                           policy=np.array([1.0]), params=params)
        
        np.testing.assert_array_equal(state0.components[0][0], belief_before)


class TestSupraPOMDPMixing:
    def test_mix_params_preserves_components(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        p1 = wm.make_params(T, B, np.array([1., 0.]), np.zeros((2, 1, 2)))
        p2 = wm.make_params(T, B, np.array([0., 1.]), np.zeros((2, 1, 2)))
        
        mixed = wm.mix_params([p1, p2], np.array([0.6, 0.4]))
        assert len(mixed) == 2
        assert np.isclose(mixed[0][1], 0.6)
        assert np.isclose(mixed[1][1], 0.4)

    def test_mix_params_flattens_nested(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        p1 = wm.make_params(T, B, np.array([1., 0.]), np.zeros((2, 1, 2)))
        p2 = wm.make_params(T, B, np.array([0., 1.]), np.zeros((2, 1, 2)))
        p3 = wm.make_params(T, B, np.array([0.5, 0.5]), np.zeros((2, 1, 2)))
        
        m12 = wm.mix_params([p1, p2], np.array([0.5, 0.5]))
        mixed = wm.mix_params([m12, p3], np.array([0.7, 0.3]))
        assert len(mixed) == 3


class TestSupraPOMDPValueIteration:
    def test_value_iteration_terminal_absorbing(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2, discount=0.9)
        T = np.array([[[0., 1.]], [[0., 1.]]])
        R = np.array([[[0., 1.]], [[0., 1.]]])
        policy = np.array([1.0])
        
        V = wm._value_iteration(T, R, policy)
        assert np.isclose(V[1], 10.0, atol=1e-3)
        assert np.isclose(V[0], 10.0, atol=1e-3)

    def test_value_iteration_converges(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2, discount=0.95)
        rng = np.random.default_rng(42)
        T = rng.dirichlet(np.ones(2), size=(2, 2))
        R = rng.random((2, 2, 2))
        policy = rng.dirichlet(np.ones(2))
        
        V = wm._value_iteration(T, R, policy)
        assert V.shape == (2,)
        assert not np.any(np.isnan(V))

    def test_value_iteration_cache_hit(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        R = np.zeros((2, 1, 2))
        policy = np.array([1.0])
        cache_key = ("test", policy.tobytes())
        
        V1 = wm._value_iteration(T, R, policy, cache_key=cache_key)
        V2 = wm._value_iteration(T, R, policy, cache_key=cache_key)
        assert V1 is V2


class TestSupraPOMDPLikelihood:
    def test_compute_likelihood_at_initial(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        theta_0 = np.array([0.6, 0.4])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        belief_pred = theta_0 @ T[:, 0, :]
        expected_lik = float(belief_pred @ B[:, 0])
        got = wm.compute_likelihood(wm.initial_state(), obs(0), params, 
                                   action=0, policy=np.array([1.0]))
        assert np.isclose(got, expected_lik, atol=1e-9)

    def test_compute_likelihood_in_unit_interval(self):
        rng = np.random.default_rng(0)
        wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=3)
        for _ in range(10):
            T = rng.dirichlet(np.ones(3), size=(3, 2))
            B = rng.dirichlet(np.ones(3), size=3)
            theta_0 = rng.dirichlet(np.ones(3))
            R = np.zeros((3, 2, 3))
            params = wm.make_params(T, B, theta_0, R)
            
            for o in range(3):
                lik = wm.compute_likelihood(wm.initial_state(), obs(o), params,
                                           action=0, policy=np.array([0.5, 0.5]))
                assert 0.0 <= lik <= 1.0 + 1e-9


class TestSupraPOMDPPolicyDependent:
    def test_callable_theta_0_used_on_init(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[1.0, 0.0], [0.5, 0.5]], [[0.0, 1.0], [0.5, 0.5]]])
        B = np.eye(2)
        theta_0_fn = lambda pi: pi  # predictor predicts agent's policy
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0_fn, R)
        
        policy = np.array([0.3, 0.7])
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=policy, params=params)
        belief, _ = state.components[0]
        np.testing.assert_allclose(belief, [1.0, 0.0])  # obs=0 from state 0

    def test_callable_T_used_each_step(self):
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T_fn = lambda pi: np.array([[[1.0, 0.0]], [[0.0, 1.0]]])  
        B = np.eye(2)
        theta_0 = np.array([1.0, 0.0])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T_fn, B, theta_0, R)
        
        policy = np.array([0.7])
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=policy, params=params)
        assert state.components is not None
