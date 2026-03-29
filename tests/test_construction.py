import pytest
from ibrl.utils.construction import parse_argument_string, construct_agent, construct_environment


class TestParseArgumentString:
    """Parametrized argument parsing tests"""
    
    @pytest.mark.parametrize("input_string,expected_name,expected_args", [
        pytest.param("classical", "classical", {}, id="no_args"),
        pytest.param("classical:epsilon=0.1", "classical", {"epsilon": 0.1}, id="single_arg"),
        pytest.param("classical:epsilon=0.1,learning_rate=0.05", "classical", 
                     {"epsilon": 0.1, "learning_rate": 0.05}, id="multi_args"),
        pytest.param("classical:epsilon=1:500:0.01", "classical", 
                     {"epsilon": (1.0, 500.0, 0.01)}, id="tuple_arg"),
        pytest.param("bayesian:temperature=1:100:0.1", "bayesian", 
                     {"temperature": (1.0, 100.0, 0.1)}, id="bayesian_tuple"),
    ])
    def test_parse_argument_string(self, input_string, expected_name, expected_args):
        name, args = parse_argument_string(input_string)
        assert name == expected_name
        assert args == expected_args


class TestConstructAgent:
    """Parametrized agent construction tests"""
    
    @pytest.mark.parametrize("agent_string,agent_type", [
        pytest.param("classical", "QLearningAgent", id="qlearn"),
        pytest.param("bayesian", "BayesianAgent", id="bayesian"),
        pytest.param("exp3", "EXP3Agent", id="exp3"),
        pytest.param("experimental1", "ExperimentalAgent1", id="exp1"),
        pytest.param("experimental2", "ExperimentalAgent2", id="exp2"),
    ])
    def test_construct_agent_types(self, agent_string, agent_type):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent(agent_string, options)
        assert agent.num_actions == 2
        assert agent.__class__.__name__ == agent_type

    @pytest.mark.parametrize("agent_string,param_name,param_value", [
        pytest.param("classical:epsilon=0.2", "epsilon", 0.2, id="qlearn_epsilon"),
        pytest.param("bayesian:temperature=1.5", "temperature", 1.5, id="bayesian_temp"),
        pytest.param("exp3:gamma=0.2", "gamma", 0.2, id="exp3_gamma"),
    ])
    def test_construct_agent_with_arguments(self, agent_string, param_name, param_value):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent(agent_string, options)
        assert getattr(agent, param_name) == param_value

    def test_invalid_agent_type(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        with pytest.raises(RuntimeError):
            construct_agent("invalid_agent", options)


class TestConstructEnvironment:
    """Parametrized environment construction tests"""
    
    @pytest.mark.parametrize("env_string,env_type", [
        pytest.param("bandit", "BanditEnvironment", id="bandit"),
        pytest.param("newcomb", "NewcombEnvironment", id="newcomb"),
        pytest.param("damascus", "DeathInDamascusEnvironment", id="damascus"),
        pytest.param("asymmetric-damascus", "AsymmetricDeathInDamascusEnvironment", id="asym_damascus"),
        pytest.param("coordination", "CoordinationGameEnvironment", id="coordination"),
        pytest.param("pdbandit", "PolicyDependentBanditEnvironment", id="pdbandit"),
    ])
    def test_construct_environment_types(self, env_string, env_type):
        options = {"num_actions": 2, "seed": 42, "verbose": 0, "num_steps": 100}
        env = construct_environment(env_string, options)
        assert env.num_actions == 2
        assert env.__class__.__name__ == env_type

    @pytest.mark.parametrize("env_string,param_name,param_value", [
        pytest.param("switching:switch_at=50", "switch_at", 50, id="switching_at"),
    ])
    def test_construct_environment_with_arguments(self, env_string, param_name, param_value):
        options = {"num_actions": 2, "seed": 42, "verbose": 0, "num_steps": 100}
        env = construct_environment(env_string, options)
        assert getattr(env, param_name) == param_value

    def test_invalid_environment_type(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        with pytest.raises(RuntimeError):
            construct_environment("invalid_env", options)
