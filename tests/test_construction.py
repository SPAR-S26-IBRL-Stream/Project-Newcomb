import pytest
from ibrl.utils.construction import parse_argument_string, construct_agent, construct_environment


class TestParseArgumentString:
    def test_no_arguments(self):
        name, args = parse_argument_string("classical")
        assert name == "classical"
        assert args == {}

    def test_single_argument(self):
        name, args = parse_argument_string("classical:epsilon=0.1")
        assert name == "classical"
        assert args == {"epsilon": 0.1}

    def test_multiple_arguments(self):
        name, args = parse_argument_string("classical:epsilon=0.1,learning_rate=0.05")
        assert name == "classical"
        assert args == {"epsilon": 0.1, "learning_rate": 0.05}

    def test_tuple_argument(self):
        name, args = parse_argument_string("classical:epsilon=1:500:0.01")
        assert name == "classical"
        assert args == {"epsilon": (1.0, 500.0, 0.01)}


class TestConstructAgent:
    def test_construct_q_learning(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("classical", options)
        assert agent.num_actions == 2

    def test_construct_bayesian(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("bayesian", options)
        assert agent.num_actions == 2

    def test_construct_with_arguments(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        agent = construct_agent("classical:epsilon=0.2", options)
        assert agent.epsilon == 0.2

    def test_invalid_agent_type(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        with pytest.raises(RuntimeError):
            construct_agent("invalid_agent", options)


class TestConstructEnvironment:
    def test_construct_bandit(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        env = construct_environment("bandit", options)
        assert env.num_actions == 2

    def test_construct_newcomb(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        env = construct_environment("newcomb", options)
        assert env.num_actions == 2

    def test_construct_with_arguments(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        env = construct_environment("switching:switch_at=50", options)
        assert env.switch_at == 50

    def test_invalid_environment_type(self):
        options = {"num_actions": 2, "seed": 42, "verbose": 0}
        with pytest.raises(RuntimeError):
            construct_environment("invalid_env", options)
