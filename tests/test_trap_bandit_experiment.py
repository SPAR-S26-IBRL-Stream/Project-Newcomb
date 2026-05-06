import numpy as np

from ibrl.agents import InfraBayesianAgent
from ibrl.environments.trap_bandit import TrapBanditEnvironment
from ibrl.exploration import Greedy, HypothesisThompsonSampling, UCB
from ibrl.infrabayesian.builders.trap_bandit import (
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_trap_bandit_hypotheses,
)
from experiments.alaro.trap_bandit.run import (
    REWARD_FUNCTION,
    TrapBanditConfig,
    bootstrap_final_regret_percentile_cis,
    get_conditions,
    sample_action_from_uniform,
    sample_world,
)


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


def test_sample_action_from_uniform_reuses_common_draw():
    assert sample_action_from_uniform(np.array([0.5, 0.5]), 0.25) == 0
    assert sample_action_from_uniform(np.array([0.5, 0.5]), 0.75) == 1
    assert sample_action_from_uniform(np.array([0.2, 0.8]), 0.25) == 1


def test_bootstrap_final_regret_percentile_cis_shapes():
    results = {
        "agent": {
            "cumulative_expected_regret": np.array([
                [0.0, 1.0],
                [0.0, 2.0],
                [0.0, 3.0],
                [0.0, 4.0],
            ])
        }
    }

    bootstrap = bootstrap_final_regret_percentile_cis(
        results,
        num_bootstrap=20,
        seed=123,
    )

    assert bootstrap["agent"]["point"].shape == (3,)
    assert bootstrap["agent"]["ci"].shape == (3, 2)
    np.testing.assert_allclose(
        bootstrap["agent"]["point"],
        np.percentile([1.0, 2.0, 3.0, 4.0], [5.0, 50.0, 95.0]),
    )


def test_uniform_p_beta_samples_valid_uniform_range():
    rng = np.random.default_rng(123)
    config = TrapBanditConfig(p_cat=0.05, p_beta=(1.0, 1.0))

    samples = [sample_world(rng, config) for _ in range(50)]

    assert all(0.0 <= sample["p1"] <= 0.95 for sample in samples)
    assert all(0.0 <= sample["p2"] <= 0.95 for sample in samples)


def test_separated_p_mode_samples_only_low_high_assignments():
    rng = np.random.default_rng(123)
    config = TrapBanditConfig(p_mode="separated", p_low=0.3, p_high=0.7)

    samples = [sample_world(rng, config) for _ in range(50)]

    assert all({sample["p1"], sample["p2"]} == {0.3, 0.7} for sample in samples)
    assert {sample["p1"] for sample in samples} == {0.3, 0.7}


def test_mostly_risky_condition_preset_uses_requested_priors():
    config = TrapBanditConfig(condition_preset="mostly_risky")

    conditions = get_conditions(config)

    assert list(conditions) == [
        "correct",
        "misspecified",
        "severely_misspecified",
        "mostly_safe_correct",
    ]
    assert conditions["correct"] == {"prior": (99.0, 1.0), "dgp": (99.0, 1.0)}
    assert conditions["misspecified"] == {"prior": (1.0, 1.0), "dgp": (99.0, 1.0)}
    assert conditions["severely_misspecified"] == {
        "prior": (1.0, 99.0),
        "dgp": (99.0, 1.0),
    }
    assert conditions["mostly_safe_correct"] == {
        "prior": (1.0, 99.0),
        "dgp": (1.0, 99.0),
    }


def test_risky_80_condition_preset_uses_requested_priors():
    config = TrapBanditConfig(condition_preset="risky_80")

    conditions = get_conditions(config)

    assert conditions["correct"] == {"prior": (4.0, 1.0), "dgp": (4.0, 1.0)}
    assert conditions["misspecified"] == {"prior": (1.0, 1.0), "dgp": (4.0, 1.0)}
    assert conditions["severely_misspecified"] == {
        "prior": (1.0, 4.0),
        "dgp": (4.0, 1.0),
    }
    assert conditions["mostly_safe_correct"] == {
        "prior": (1.0, 4.0),
        "dgp": (1.0, 4.0),
    }


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


def test_trap_bandit_hypothesis_builder_accepts_separated_pairs():
    _wm, safe, risky = make_trap_bandit_hypotheses(
        p_cat=0.01,
        p_pairs=[(0.3, 0.7), (0.7, 0.3)],
    )

    assert len(safe.measures[0].params.components) == 2
    assert len(risky.measures[0].params.components) == 2
    safe_pairs = {
        tuple(component.metadata["p_values"])
        for component in safe.measures[0].params.components
    }
    risky_trapped = {
        component.metadata["trapped_arm"]
        for component in risky.measures[0].params.components
    }
    assert safe_pairs == {(0.3, 0.7), (0.7, 0.3)}
    assert risky_trapped == {0, 1}


def test_thompson_sampling_returns_valid_policy():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    bayes_h = make_bayesian_hypothesis(safe, risky, alpha_beta=(2.0, 2.0))
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=HypothesisThompsonSampling(),
        epsilon=0.0,
    )
    agent.reset()
    probs = agent.get_probabilities()
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)
