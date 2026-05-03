import numpy as np

from ibrl.agents import Greedy, InfraBayesianAgent, ThompsonSampling, UCB
from ibrl.environments.trap_bandit import TrapBanditEnvironment
from ibrl.infrabayesian.builders.trap_bandit import (
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_trap_bandit_hypotheses,
)
from experiments.alaro.trap_bandit.run import REWARD_FUNCTION


def test_trap_bandit_environment_marks_catastrophe_observation():
    env = TrapBanditEnvironment(
        p1=0.5,
        p2=0.2,
        risky=True,
        p_cat=1.0,
        seed=123,
    )
    env.reset()
    outcome = env.step(np.array([1.0, 0.0]), 0)
    assert outcome.reward == -1000.0
    assert outcome.observation == 2


def test_greedy_tie_breaks_uniformly():
    strategy = Greedy()
    class Agent:
        num_actions = 2
    probs = strategy.get_probabilities(Agent(), np.array([1.0, 1.0]))
    np.testing.assert_allclose(probs, [0.5, 0.5])


def test_ucb_tries_unpulled_actions():
    class Agent:
        num_actions = 2
        action_counts = np.array([10, 0])
        empirical_values = np.array([1.0, 0.0])
        step = 11
    probs = UCB().get_probabilities(Agent(), np.array([1.0, 0.0]))
    np.testing.assert_allclose(probs, [0.0, 1.0])


def test_trap_bandit_hypothesis_builders_construct_agents():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    bayes_h = make_bayesian_hypothesis(safe, risky, alpha_beta=(2.0, 2.0))
    ib_h = make_ib_hypothesis(safe, risky)

    bayes = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=Greedy(),
        epsilon=0.0,
    )
    ib = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[ib_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=Greedy(),
        epsilon=0.0,
    )
    bayes.reset()
    ib.reset()
    assert bayes.get_probabilities().shape == (2,)
    assert ib.get_probabilities().shape == (2,)


def test_thompson_sampling_returns_valid_policy():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    bayes_h = make_bayesian_hypothesis(safe, risky, alpha_beta=(2.0, 2.0))
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=ThompsonSampling(),
        epsilon=0.0,
    )
    agent.reset()
    probs = agent.get_probabilities()
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)
