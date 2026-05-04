"""Smoke tests: verify every SupraPOMDPWorldModel addition works end-to-end."""
import numpy as np
import pytest

from ibrl.outcome import Outcome
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_models.supra_pomdp_world_model import (
    SupraPOMDPWorldModel, SupraPOMDPWorldModelBeliefState
)
from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.simulators import simulate


def obs(o: int, reward: float = 0.) -> Outcome:
    """Helper: create Outcome with observation field."""
    return Outcome(reward=reward, observation=o)


class TestOutcomeObservationField:
    """Smoke: Outcome.observation field exists and works."""
    
    def test_outcome_has_observation_field(self):
        """Outcome can be constructed with observation parameter."""
        outcome = Outcome(reward=1.0, observation=0)
        assert outcome.observation == 0
        assert outcome.reward == 1.0
    
    def test_outcome_observation_none_by_default(self):
        """Outcome.observation defaults to None."""
        outcome = Outcome(reward=1.0)
        assert outcome.observation is None
    
    def test_outcome_backward_compat_env_action(self):
        """Outcome still accepts env_action for backward compatibility."""
        outcome = Outcome(reward=1.0, env_action=5)
        assert outcome.env_action == 5


class TestSupraPOMDPWorldModelImport:
    """Smoke: SupraPOMDPWorldModel can be imported and instantiated."""
    
    def test_import_from_world_models_package(self):
        """SupraPOMDPWorldModel is exported from world_models package."""
        from ibrl.infrabayesian.world_models import SupraPOMDPWorldModel as WM
        assert WM is not None
    
    def test_import_from_direct_module(self):
        """SupraPOMDPWorldModel can be imported directly."""
        from ibrl.infrabayesian.world_models.supra_pomdp_world_model import SupraPOMDPWorldModel
        assert SupraPOMDPWorldModel is not None
    
    def test_instantiate_with_defaults(self):
        """SupraPOMDPWorldModel instantiates with default hyperparameters."""
        wm = SupraPOMDPWorldModel(num_states=4, num_actions=2, num_obs=3)
        assert wm.num_states == 4
        assert wm.num_actions == 2
        assert wm.num_obs == 3
        assert wm.discount == 0.95
        assert wm.value_iter_tol == 1e-6
        assert wm.value_iter_max == 1000
    
    def test_instantiate_with_custom_hyperparameters(self):
        """SupraPOMDPWorldModel accepts custom hyperparameters."""
        wm = SupraPOMDPWorldModel(num_states=8, num_actions=4, num_obs=6,
                                   discount=0.99, value_iter_tol=1e-8,
                                   value_iter_max=2000)
        assert wm.discount == 0.99
        assert wm.value_iter_tol == 1e-8
        assert wm.value_iter_max == 2000


class TestSupraPOMDPMakeParams:
    """Smoke: make_params validates and bundles POMDP parameters."""
    
    def test_make_params_static_arrays(self):
        """make_params accepts static T, B, theta_0, R arrays."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.7, 0.3])
        R = np.random.randn(2, 2, 2)
        
        params = wm.make_params(T, B, theta_0, R)
        assert params == (T, B, theta_0, R)
    
    def test_make_params_callable_theta_0(self):
        """make_params accepts callable theta_0."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0_fn = lambda pi: pi / pi.sum()
        R = np.zeros((2, 2, 2))
        
        params = wm.make_params(T, B, theta_0_fn, R)
        assert callable(params[2])
    
    def test_make_params_callable_T(self):
        """make_params accepts callable T."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T_fn = lambda pi: np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        
        params = wm.make_params(T_fn, B, theta_0, R)
        assert callable(params[0])
    
    def test_make_params_callable_B(self):
        """make_params accepts callable B."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B_fn = lambda pi: np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        
        params = wm.make_params(T, B_fn, theta_0, R)
        assert callable(params[1])
    
    def test_make_params_rejects_invalid_T_shape(self):
        """make_params rejects T with wrong shape."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        bad_T = np.ones((3, 2, 2))  # wrong first dimension
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        
        with pytest.raises(AssertionError):
            wm.make_params(bad_T, B, theta_0, R)
    
    def test_make_params_rejects_nonstochastic_T(self):
        """make_params rejects T that doesn't sum to 1."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        bad_T = np.ones((2, 2, 2))  # sums to 2, not 1
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        
        with pytest.raises(AssertionError):
            wm.make_params(bad_T, B, theta_0, R)


class TestSupraPOMDPEventIndex:
    """Smoke: event_index extracts observation from Outcome."""
    
    def test_event_index_returns_observation(self):
        """event_index returns outcome.observation."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=3)
        outcome = obs(2, reward=1.0)
        idx = wm.event_index(outcome, action=0)
        assert idx == 2
    
    def test_event_index_with_zero_observation(self):
        """event_index works with observation=0."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=3)
        outcome = obs(0, reward=0.5)
        idx = wm.event_index(outcome, action=1)
        assert idx == 0


class TestSupraPOMDPInitialState:
    """Smoke: initial_state and is_initial work correctly."""
    
    def test_initial_state_returns_belief_state_object(self):
        """initial_state returns SupraPOMDPWorldModelBeliefState."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        state = wm.initial_state()
        assert isinstance(state, SupraPOMDPWorldModelBeliefState)
    
    def test_initial_state_components_is_none(self):
        """initial_state has components=None."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        state = wm.initial_state()
        assert state.components is None
    
    def test_is_initial_recognizes_none_state(self):
        """is_initial returns True for None components."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        state = wm.initial_state()
        assert wm.is_initial(state)
    
    def test_is_initial_recognizes_non_initial_state(self):
        """is_initial returns False for non-None components."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        belief = np.array([0.5, 0.5])
        state = SupraPOMDPWorldModelBeliefState([(belief, 0.0)])
        assert not wm.is_initial(state)


class TestSupraPOMDPMixParams:
    """Smoke: mix_params combines multiple hypotheses."""
    
    def test_mix_params_single_component(self):
        """mix_params with single component returns list with one element."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        mixed = wm.mix_params([params], np.array([1.0]))
        assert len(mixed) == 1
        assert np.isclose(mixed[0][1], 1.0)
    
    def test_mix_params_multiple_components(self):
        """mix_params with multiple components returns list with correct weights."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0_1 = np.array([0.7, 0.3])
        theta_0_2 = np.array([0.3, 0.7])
        R = np.zeros((2, 2, 2))
        
        p1 = wm.make_params(T, B, theta_0_1, R)
        p2 = wm.make_params(T, B, theta_0_2, R)
        
        mixed = wm.mix_params([p1, p2], np.array([0.6, 0.4]))
        assert len(mixed) == 2
        assert np.isclose(mixed[0][1], 0.6)
        assert np.isclose(mixed[1][1], 0.4)
    
    def test_mix_params_flattens_nested_mixtures(self):
        """mix_params flattens nested mixture lists."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        
        p1 = wm.make_params(T, B, theta_0, R)
        p2 = wm.make_params(T, B, theta_0, R)
        p3 = wm.make_params(T, B, theta_0, R)
        
        nested = [(p1, 0.5), (p2, 0.5)]
        mixed = wm.mix_params([nested, p3], np.array([0.7, 0.3]))
        assert len(mixed) == 3


class TestSupraPOMDPUpdateState:
    """Smoke: update_state performs Bayesian filtering."""
    
    def test_update_state_lazy_initializes(self):
        """update_state initializes belief from theta_0 on first call."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        theta_0 = np.array([0.8, 0.2])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=np.array([1.0]), params=params)
        assert state.components is not None
        assert len(state.components) == 1
        belief, log_m = state.components[0]
        assert belief.shape == (2,)
        assert np.isfinite(log_m)
    
    def test_update_state_returns_belief_state_object(self):
        """update_state returns SupraPOMDPWorldModelBeliefState."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=np.array([1.0]), params=params)
        assert isinstance(state, SupraPOMDPWorldModelBeliefState)
    
    def test_update_state_with_mixture(self):
        """update_state handles mixed params (list of components)."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.eye(2)
        R = np.zeros((2, 1, 2))
        
        p1 = wm.make_params(T, B, np.array([0.7, 0.3]), R)
        p2 = wm.make_params(T, B, np.array([0.3, 0.7]), R)
        mixed = [(p1, 0.5), (p2, 0.5)]
        
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=np.array([1.0]), params=mixed)
        assert len(state.components) == 2


class TestSupraPOMDPComputeLikelihood:
    """Smoke: compute_likelihood returns valid probability."""
    
    def test_compute_likelihood_returns_scalar(self):
        """compute_likelihood returns a float."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        theta_0 = np.array([0.6, 0.4])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        lik = wm.compute_likelihood(wm.initial_state(), obs(0), params,
                                   action=0, policy=np.array([1.0]))
        assert isinstance(lik, (float, np.floating))
    
    def test_compute_likelihood_in_unit_interval(self):
        """compute_likelihood returns value in [0, 1]."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.array([[0.9, 0.1], [0.2, 0.8]])
        theta_0 = np.array([0.6, 0.4])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        lik = wm.compute_likelihood(wm.initial_state(), obs(0), params,
                                   action=0, policy=np.array([1.0]))
        assert 0.0 <= lik <= 1.0


class TestSupraPOMDPComputeExpectedReward:
    """Smoke: compute_expected_reward returns valid value."""
    
    def test_compute_expected_reward_returns_scalar(self):
        """compute_expected_reward returns a float."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.random.randn(2, 1, 2)
        params = wm.make_params(T, B, theta_0, R)
        
        reward_fn = np.array([0., 1.])
        er = wm.compute_expected_reward(wm.initial_state(), reward_fn, params,
                                       action=0, policy=np.array([1.0]))
        assert isinstance(er, (float, np.floating))
    
    def test_compute_expected_reward_with_mixture(self):
        """compute_expected_reward handles mixed params."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.eye(2)
        R = np.random.randn(2, 1, 2)
        
        p1 = wm.make_params(T, B, np.array([0.7, 0.3]), R)
        p2 = wm.make_params(T, B, np.array([0.3, 0.7]), R)
        mixed = [(p1, 0.5), (p2, 0.5)]
        
        reward_fn = np.array([0., 1.])
        er = wm.compute_expected_reward(wm.initial_state(), reward_fn, mixed,
                                       action=0, policy=np.array([1.0]))
        assert isinstance(er, (float, np.floating))


class TestSupraPOMDPValueIteration:
    """Smoke: _value_iteration converges and caches."""
    
    def test_value_iteration_returns_vector(self):
        """_value_iteration returns array of shape (num_states,)."""
        wm = SupraPOMDPWorldModel(num_states=3, num_actions=2, num_obs=2)  # ADD num_obs=2
        T = np.random.dirichlet(np.ones(3), size=(3, 2))
        R = np.random.randn(3, 2, 3)
        policy = np.array([0.5, 0.5])
        
        V = wm._value_iteration(T, R, policy)
        assert V.shape == (3,)
        assert np.all(np.isfinite(V))
    
    def test_value_iteration_caches_result(self):
        """_value_iteration caches and returns same object on cache hit."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)  # ADD num_obs=2
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        R = np.zeros((2, 1, 2))
        policy = np.array([1.0])
        cache_key = ("test", policy.tobytes())
        
        V1 = wm._value_iteration(T, R, policy, cache_key=cache_key)
        V2 = wm._value_iteration(T, R, policy, cache_key=cache_key)
        assert V1 is V2

class TestSupraPOMDPWithInfradistribution:
    """Smoke: SupraPOMDPWorldModel integrates with Infradistribution."""
    
    def test_create_infradistribution_with_supra_pomdp(self):
        """Infradistribution can be created with SupraPOMDPWorldModel."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        dist = Infradistribution([AMeasure(params)], world_model=wm)
        assert dist.world_model is wm
        assert len(dist.measures) == 1
    
    def test_mix_infradistributions_with_supra_pomdp(self):
        """Infradistribution.mix works with SupraPOMDPWorldModel."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        R = np.zeros((2, 2, 2))
        
        p1 = wm.make_params(T, B, np.array([0.7, 0.3]), R)
        p2 = wm.make_params(T, B, np.array([0.3, 0.7]), R)
        
        d1 = Infradistribution([AMeasure(p1)], world_model=wm)
        d2 = Infradistribution([AMeasure(p2)], world_model=wm)
        
        mixed = Infradistribution.mix([d1, d2], np.array([0.5, 0.5]))
        assert mixed.world_model is wm
    
    def test_evaluate_action_with_supra_pomdp(self):
        """Infradistribution.evaluate_action works with SupraPOMDPWorldModel."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        dist = Infradistribution([AMeasure(params)], world_model=wm)
        reward_fn = np.array([0., 1.])
        
        val = dist.evaluate_action(reward_fn, action=0, policy=np.array([0.5, 0.5]))
        assert isinstance(val, (float, np.floating))
        assert np.isfinite(val)
    
    def test_update_infradistribution_with_supra_pomdp(self):
        """Infradistribution.update works with SupraPOMDPWorldModel."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        dist = Infradistribution([AMeasure(params)], world_model=wm)
        reward_fn = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        # Use valid policy that doesn't create zero probability
        policy = np.array([0.5, 0.5])
        dist.update(reward_fn, obs(0), action=0, policy=policy)
        # Belief may or may not update depending on probability threshold
        # Just verify no crash occurs
        assert isinstance(dist.belief_state, SupraPOMDPWorldModelBeliefState)


class TestSupraPOMDPEndToEndSimulation:
    """Smoke: Full simulation loop with SupraPOMDPWorldModel."""
    
    def test_simulate_with_supra_pomdp_agent(self):
        """Full simulate() loop works with SupraPOMDPWorldModel agent."""
        from ibrl.environments.base import BaseEnvironment
        
        class DummyEnvironment(BaseEnvironment):
            """Minimal environment for testing."""
            def _resolve(self, observation, action):
                return 0.5
            
            def get_optimal_reward(self):
                return 1.0
        
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        hypothesis = Infradistribution([AMeasure(params)], world_model=wm)
        agent = InfraBayesianAgent(
            num_actions=2,
            hypotheses=[hypothesis],
            prior=np.array([1.0]),
            reward_function=np.array([[0.5, 0.5], [0.5, 0.5]]),
            seed=42
        )
        
        env = DummyEnvironment(num_actions=2, seed=43)
        options = {"num_steps": 3, "num_runs": 1}
        
        results = simulate(env, agent, options)
        assert results is not None
        assert "average_reward" in results


class TestSupraPOMDPPolicyDependence:
    """Smoke: Policy-dependent kernels work correctly."""
    
    def test_policy_dependent_theta_0_resolves_correctly(self):
        """Policy-dependent theta_0 is resolved with current policy."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        theta_0_fn = lambda pi: pi  # predictor predicts agent's policy
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B, theta_0_fn, R)
        
        policy = np.array([0.3, 0.7])
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=policy, params=params)
        belief, _ = state.components[0]
        # Initial belief should be conditioned on obs=0 from policy [0.3, 0.7]
        # B=eye(2), so obs=0 means state=0 with certainty
        np.testing.assert_allclose(belief, [1.0, 0.0])
    
    def test_policy_dependent_T_resolves_correctly(self):
        """Policy-dependent T is resolved with current policy."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T_fn = lambda pi: np.array([[[pi[0], pi[1]], [0.5, 0.5]],
                                     [[pi[1], pi[0]], [0.5, 0.5]]])
        B = np.eye(2)
        theta_0 = np.array([1.0, 0.0])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T_fn, B, theta_0, R)
        
        policy = np.array([0.6, 0.4])
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=policy, params=params)
        assert state.components is not None
    
    def test_policy_dependent_B_resolves_correctly(self):
        """Policy-dependent B is resolved with current policy."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B_fn = lambda pi: np.array([[pi[0], pi[1]], [pi[1], pi[0]]])
        theta_0 = np.array([0.5, 0.5])
        R = np.zeros((2, 2, 2))
        params = wm.make_params(T, B_fn, theta_0, R)
        
        policy = np.array([0.7, 0.3])
        state = wm.update_state(wm.initial_state(), obs(0), action=0,
                               policy=policy, params=params)
        assert state.components is not None


class TestSupraPOMDPResolveHelper:
    """Smoke: _resolve helper works for both static and callable kernels."""
    
    def test_resolve_static_array(self):
        """_resolve returns static array unchanged."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        arr = np.array([[1, 2], [3, 4]])
        policy = np.array([0.5, 0.5])
        
        result = wm._resolve(arr, policy)
        np.testing.assert_array_equal(result, arr)
    
    def test_resolve_callable(self):
        """_resolve calls callable with policy."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        fn = lambda pi: pi * 2
        policy = np.array([0.5, 0.5])
        
        result = wm._resolve(fn, policy)
        np.testing.assert_array_equal(result, np.array([1.0, 1.0]))


class TestSupraPOMDPBackwardCompat:
    """Smoke: Backward compatibility maintained."""
    
    def test_world_model_import_from_shim(self):
        """WorldModel can still be imported from world_model.py shim."""
        from ibrl.infrabayesian.world_model import WorldModel
        assert WorldModel is not None
    
    def test_outcome_env_action_still_works(self):
        """Outcome.env_action field still works for backward compat."""
        outcome = Outcome(reward=1.0, env_action=5)
        assert outcome.env_action == 5
        assert outcome.reward == 1.0
    
    def test_outcome_both_fields_work(self):
        """Outcome can have both observation and env_action."""
        outcome = Outcome(reward=1.0, observation=2, env_action=5)
        assert outcome.observation == 2
        assert outcome.env_action == 5


class TestSupraPOMDPMultipleHypotheses:
    """Smoke: Multiple hypotheses with credal mixture."""
    
    def test_credal_mixture_of_supra_pomdps(self):
        """Credal mixture of multiple SupraPOMDP hypotheses works."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        R = np.zeros((2, 2, 2))
        
        p1 = wm.make_params(T, B, np.array([0.8, 0.2]), R)
        p2 = wm.make_params(T, B, np.array([0.2, 0.8]), R)
        p3 = wm.make_params(T, B, np.array([0.5, 0.5]), R)
        
        d1 = Infradistribution([AMeasure(p1)], world_model=wm)
        d2 = Infradistribution([AMeasure(p2)], world_model=wm)
        d3 = Infradistribution([AMeasure(p3)], world_model=wm)
        
        mixed = Infradistribution.mixKU([d1, d2, d3])
        assert len(mixed.measures) == 3
    
    def test_agent_with_multiple_supra_pomdp_hypotheses(self):
        """Agent with multiple SupraPOMDP hypotheses works."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=2, num_obs=2)
        T = np.array([[[0.9, 0.1], [0.5, 0.5]], [[0.5, 0.5], [0.1, 0.9]]])
        B = np.eye(2)
        R = np.zeros((2, 2, 2))
        
        p1 = wm.make_params(T, B, np.array([0.8, 0.2]), R)
        p2 = wm.make_params(T, B, np.array([0.2, 0.8]), R)
        
        h1 = Infradistribution([AMeasure(p1)], world_model=wm)
        h2 = Infradistribution([AMeasure(p2)], world_model=wm)
        
        agent = InfraBayesianAgent(
            num_actions=2,
            hypotheses=[h1, h2],
            prior=np.array([0.5, 0.5]),
            reward_function=np.ones((2, 2))
        )
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (2,)
        assert np.isclose(probs.sum(), 1.0)


class TestSupraPOMDPNumericalStability:
    """Smoke: Numerical stability checks."""
    
    def test_zero_likelihood_handled_gracefully(self):
        """Zero-likelihood observations don't crash."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2)
        T = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
        B = np.array([[1.0, 0.0], [0.0, 1.0]])  # deterministic
        theta_0 = np.array([1.0, 0.0])
        R = np.zeros((2, 1, 2))
        params = wm.make_params(T, B, theta_0, R)
        
        # obs=1 has zero probability from state 0
        state = wm.update_state(wm.initial_state(), obs(1), action=0,
                               policy=np.array([1.0]), params=params)
        assert state.components is not None
        assert not np.any(np.isnan(state.components[0][0]))
    
    def test_large_discount_factor(self):
        """Large discount factor (close to 1) handled correctly."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2,
                                   discount=0.999)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.random.randn(2, 1, 2)
        params = wm.make_params(T, B, theta_0, R)
        
        V = wm._value_iteration(T, R, np.array([1.0]))
        assert np.all(np.isfinite(V))
    
    def test_small_value_iter_tolerance(self):
        """Small value iteration tolerance converges."""
        wm = SupraPOMDPWorldModel(num_states=2, num_actions=1, num_obs=2,  # ADD num_obs=2
                                   value_iter_tol=1e-10)
        T = np.array([[[0.8, 0.2]], [[0.3, 0.7]]])
        B = np.eye(2)
        theta_0 = np.array([0.5, 0.5])
        R = np.random.randn(2, 1, 2)
        params = wm.make_params(T, B, theta_0, R)
        
        V = wm._value_iteration(T, R, np.array([1.0]))
        assert np.all(np.isfinite(V))
        
