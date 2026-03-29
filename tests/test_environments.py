import numpy as np
import pytest
from ibrl.environments import (
    BanditEnvironment,
    NewcombEnvironment,
    DeathInDamascusEnvironment,
    CoordinationGameEnvironment,
    PolicyDependentBanditEnvironment,
    AsymmetricDeathInDamascusEnvironment,
    SwitchingAdversaryEnvironment,
    MatchEnvironment,
    ReverseTailsEnvironment
)


class TestEnvironmentInitialization:
    """Parametrized environment initialization"""
    
    @pytest.mark.parametrize("env_class,num_actions", [
        pytest.param(BanditEnvironment, 2, id="bandit"),
        pytest.param(NewcombEnvironment, 2, id="newcomb"),
        pytest.param(DeathInDamascusEnvironment, 2, id="damascus"),
        pytest.param(CoordinationGameEnvironment, 2, id="coordination"),
        pytest.param(PolicyDependentBanditEnvironment, 2, id="pdbandit"),
        pytest.param(AsymmetricDeathInDamascusEnvironment, 2, id="asym_damascus"),
        pytest.param(SwitchingAdversaryEnvironment, 2, id="switching"),
        pytest.param(MatchEnvironment, 2, id="match"),
        
        pytest.param(ReverseTailsEnvironment, 2, id="reverse_tails"),
    ])
    def test_initialization(self, env_class, num_actions, seed):
        if env_class == SwitchingAdversaryEnvironment:
            env = env_class(num_actions=num_actions, num_steps=100, seed=seed)
        else:
            env = env_class(num_actions=num_actions, seed=seed)
        assert env.num_actions == num_actions

    @pytest.mark.parametrize("env_class,num_actions", [
        pytest.param(BanditEnvironment, 2, id="bandit"),
        pytest.param(NewcombEnvironment, 2, id="newcomb"),
        pytest.param(DeathInDamascusEnvironment, 2, id="damascus"),
    ])
    def test_reset(self, env_class, num_actions, seed):
        env = env_class(num_actions=num_actions, seed=seed)
        env.reset()
        assert hasattr(env, 'random')


class TestRewardTableStructure:
    """Parametrized reward table tests"""
    
    @pytest.mark.parametrize("env_class,expected_table", [
        pytest.param(NewcombEnvironment, [[10, 15], [0, 5]], id="newcomb"),
        pytest.param(DeathInDamascusEnvironment, [[0, 10], [10, 0]], id="damascus"),
        pytest.param(CoordinationGameEnvironment, [[2, 0], [0, 1]], id="coordination"),
        pytest.param(MatchEnvironment, [[1.0, 0.0], [0.0, 1.0]], id="match"),
        pytest.param(ReverseTailsEnvironment, [[0.0, 1.0], [0.5, 0.5]], id="reverse_tails"),
    ])
    def test_reward_table_structure(self, env_class, expected_table, seed):
        env = env_class(num_actions=2, seed=seed)
        assert np.allclose(env.reward_table, expected_table)


class TestBanditEnvironment:
    def test_reward_is_always_finite(self):
        """Rewards should never be NaN or Inf"""
        env = BanditEnvironment(num_actions=2, seed=42)
        for _ in range(100):
            env.reset()
            reward = env.interact(0)
            assert np.isfinite(reward), f"Non-finite reward: {reward}"

    def test_environment_rejects_invalid_actions(self):
        """Environment should reject actions outside range"""
        env = BanditEnvironment(num_actions=2, seed=42)
        env.reset()
        with pytest.raises((IndexError, AssertionError)):
            env.interact(5)


class TestNewcombEnvironment:
    def test_predict_sets_rewards(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        newcomb_env.predict(probs)
        assert hasattr(newcomb_env, 'rewards')
        assert newcomb_env.rewards.shape == (2,)

    def test_interact(self, newcomb_env):
        probs = np.array([1.0, 0.0])
        newcomb_env.predict(probs)
        reward = newcomb_env.interact(0)
        assert isinstance(reward, (int, float, np.integer, np.floating))
        assert reward in [0, 5, 10, 15]

    def test_get_optimal_reward(self, newcomb_env):
        optimal = newcomb_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert optimal >= 0


class TestDeathInDamascusEnvironment:
    def test_get_optimal_reward(self, damascus_env):
        optimal = damascus_env.get_optimal_reward()
        assert isinstance(optimal, (int, float, np.integer, np.floating))
        assert 0 <= optimal <= 10


class TestCoordinationGameEnvironment:
    def test_custom_rewards(self, seed):
        env = CoordinationGameEnvironment(num_actions=2, rewardA=5, rewardB=3, seed=seed)
        assert env.reward_table[0, 0] == 5
        assert env.reward_table[1, 1] == 3


class TestPolicyDependentBanditEnvironment:
    def test_random_reward_table(self):
        """Test that reward table is randomized with different seeds"""
        env1 = PolicyDependentBanditEnvironment(num_actions=2, seed=42)
        env1.reset()
        
        env2 = PolicyDependentBanditEnvironment(num_actions=2, seed=43)
        env2.reset()
        
        assert not np.allclose(env1.reward_table, env2.reward_table)

    def test_predict_and_interact(self, pdbandit_env):
        probs = np.array([1.0, 0.0])
        pdbandit_env.predict(probs)
        reward = pdbandit_env.interact(0)
        assert isinstance(reward, (int, float, np.floating))


class TestAsymmetricDeathInDamascusEnvironment:
    def test_default_rewards(self, asymmetric_damascus_env):
        assert asymmetric_damascus_env.reward_table[0, 0] == 0
        assert asymmetric_damascus_env.reward_table[1, 1] == 5
        assert asymmetric_damascus_env.reward_table[0, 1] == 10
        assert asymmetric_damascus_env.reward_table[1, 0] == 10


class TestSwitchingAdversaryEnvironment:
    def test_switch_at_parameter(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=50, seed=seed)
        assert env.switch_at == 50

    def test_default_switch_at(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, num_steps=100, seed=seed)
        assert env.switch_at == 50

    def test_reset(self, switching_env):
        switching_env.reset()
        assert switching_env.step == 0
        assert switching_env.values[0] == 1.0
        assert switching_env.values[1] == 0.0

    def test_interact_before_switch(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=50, seed=seed)
        env.reset()
        reward = env.interact(0)
        assert isinstance(reward, (int, float, np.floating))
        assert env.step == 1

    def test_interact_after_switch(self, seed):
        env = SwitchingAdversaryEnvironment(num_actions=2, switch_at=2, seed=seed)
        env.reset()
        
        env.interact(0)
        env.interact(0)
        env.interact(0)
        assert env.values[-1] == 1.0
        assert env.values[0] == 0.0

    def test_get_optimal_reward(self, switching_env):
        optimal = switching_env.get_optimal_reward()
        assert optimal == 1.0

    def test_step_increments(self, switching_env):
        switching_env.reset()
        assert switching_env.step == 0
        
        switching_env.interact(0)
        assert switching_env.step == 1
        
        switching_env.interact(1)
        assert switching_env.step == 2


class TestMatchEnvironment:
    def test_predict_and_interact(self, match_env):
        probs = np.array([1.0, 0.0])
        match_env.predict(probs)
        reward = match_env.interact(0)
        assert reward == 1.0

    def test_mismatch_reward(self, match_env):
        """Test mismatch gives 0 reward"""
        probs = np.array([1.0, 0.0])
        match_env.predict(probs)
        reward = match_env.interact(1)
        assert reward == 0.0


class TestReverseTailsEnvironment:
    def test_heads_mismatch_reward(self, reverse_tails_env):
        """Test mismatch on heads gives 1.0"""
        probs = np.array([1.0, 0.0])
        reverse_tails_env.predict(probs)
        reward = reverse_tails_env.interact(1)
        assert reward == 1.0

    def test_tails_always_half(self, reverse_tails_env):
        """Test tails always gives 0.5 regardless of action"""
        probs = np.array([0.0, 1.0])
        reverse_tails_env.predict(probs)
        
        reward_action0 = reverse_tails_env.interact(0)
        assert reward_action0 == 0.5
        
        reverse_tails_env.reset()
        reverse_tails_env.predict(probs)
        reward_action1 = reverse_tails_env.interact(1)
        assert reward_action1 == 0.5

    def test_asymmetric_behavior(self, reverse_tails_env):
        """Test that heads and tails have different reward structures"""
        assert reverse_tails_env.reward_table[0, 1] > reverse_tails_env.reward_table[0, 0]
        assert reverse_tails_env.reward_table[1, 0] == reverse_tails_env.reward_table[1, 1]


class TestNewcombLikeEnvironments:
    """Test Newcomb-like environment behavior"""
    
    @pytest.mark.parametrize("probs,action,expected_reward", [
        pytest.param([1.0, 0.0], 0, 10, id="newcomb_pred0_act0"),
        pytest.param([1.0, 0.0], 1, 15, id="newcomb_pred0_act1"),
        pytest.param([0.0, 1.0], 0, 0, id="newcomb_pred1_act0"),
        pytest.param([0.0, 1.0], 1, 5, id="newcomb_pred1_act1"),
    ])
    def test_newcomb_reward_table_consistency(self, probs, action, expected_reward, seed):
        """Newcomb environment should return correct rewards"""
        env = NewcombEnvironment(num_actions=2, seed=seed)
        env.reset()
        env.predict(np.array(probs))
        reward = env.interact(action)
        assert reward == expected_reward

    @pytest.mark.parametrize("env_class,probs,action,expected_reward", [
        pytest.param(MatchEnvironment, [1.0, 0.0], 0, 1.0, id="match_heads_match"),
        pytest.param(MatchEnvironment, [1.0, 0.0], 1, 0.0, id="match_heads_mismatch"),
        pytest.param(ReverseTailsEnvironment, [1.0, 0.0], 1, 1.0, id="reverse_heads_mismatch"),
        pytest.param(ReverseTailsEnvironment, [0.0, 1.0], 0, 0.5, id="reverse_tails_action0"),
    ])
    def test_match_environment_rewards(self, env_class, probs, action, expected_reward, seed):
        """Test Match environment rewards"""
        env = env_class(num_actions=2, seed=seed)
        env.reset()
        env.predict(np.array(probs))
        reward = env.interact(action)
        assert reward == expected_reward
