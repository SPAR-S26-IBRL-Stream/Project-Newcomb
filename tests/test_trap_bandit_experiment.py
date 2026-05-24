import numpy as np
import pytest

from ibrl.agents import InfraBayesianAgent
from ibrl.environments.trap_bandit import TrapBanditEnvironment
from ibrl.exploration import BayesianUCB, Greedy, HypothesisThompsonSampling
from ibrl.infrabayesian.builders.trap_bandit import (
    make_bayesian_hypothesis,
    make_ib_hypothesis,
    make_trap_bandit_hypotheses,
)
from experiments.alaro.trap_bandit.run import (
    REPORT_FIGURES,
    REWARD_FUNCTION,
    TrapBanditConfig,
    bootstrap_final_regret_percentile_cis,
    parse_args,
    report_conditions,
    run_condition,
    sample_world,
    summarize,
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


def test_sample_world_uses_point_p_risky():
    rng = np.random.default_rng(123)
    never_risky = TrapBanditConfig(p_risky=0.0)
    always_risky = TrapBanditConfig(p_risky=1.0)

    assert not any(sample_world(rng, never_risky)["risky"] for _ in range(20))
    assert all(sample_world(rng, always_risky)["risky"] for _ in range(20))


def test_sample_world_uses_separated_low_high_assignments():
    rng = np.random.default_rng(123)
    config = TrapBanditConfig(p_low=0.3, p_high=0.7)

    samples = [sample_world(rng, config) for _ in range(50)]

    assert all({sample["p1"], sample["p2"]} == {0.3, 0.7} for sample in samples)
    assert {sample["p1"] for sample in samples} == {0.3, 0.7}


def test_cli_defaults_match_report_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run.py"])

    args = parse_args()

    assert args.num_worlds == 200
    assert args.num_steps == 100
    assert args.p_cat == 0.01
    assert args.p_low == 0.3
    assert args.p_high == 0.7
    assert args.bootstrap_samples == 5000
    assert args.output_dir.name == "results_report_200_pcat001"


def test_report_figures_match_results_md_design():
    assert [figure.name for figure in REPORT_FIGURES] == [
        "mostly_risky",
        "mostly_safe",
        "balanced",
    ]

    pairs = {
        figure.name: [
            (condition.p_risky, condition.p_risky_prior)
            for condition in figure.conditions
        ]
        for figure in REPORT_FIGURES
    }
    assert pairs == {
        "mostly_risky": [(0.99, 0.99), (0.99, 0.01)],
        "mostly_safe": [(0.01, 0.01), (0.01, 0.99)],
        "balanced": [(0.5, 0.5), (0.5, 0.99)],
    }


def test_report_condition_names_are_unique():
    conditions = report_conditions()

    assert len(conditions) == 6
    assert set(conditions) == {
        "mostly_risky_correct",
        "mostly_risky_severely_misspecified",
        "mostly_safe_correct",
        "mostly_safe_severely_pessimistic",
        "balanced_correct",
        "balanced_severely_pessimistic",
    }

def test_run_condition_tiny_config_end_to_end():
    config = TrapBanditConfig(
        num_worlds=2,
        num_steps=10,
        p_low=0.3,
        p_high=0.7,
    )

    results = run_condition(0.5, config, kinds=["bayes_greedy", "ib"])
    summary = summarize(results)

    assert set(results) == {"bayes_greedy", "ib"}
    assert all(len(runs) == 2 for runs in results.values())
    for kind in results:
        assert len(results[kind][0]["cumulative_expected_regret"]) == 10
        assert summary[kind]["all"]["regret_p5_p50_p95"].shape == (3, 10)


def test_bayesian_ucb_returns_valid_policy():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    bayes_h = make_bayesian_hypothesis(safe, risky, p_risky=0.5)
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=BayesianUCB(quantile=0.95),
    )
    agent.reset()
    probs = agent.get_probabilities()
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)


def test_trap_bandit_hypothesis_builders_construct_agents():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    bayes_h = make_bayesian_hypothesis(safe, risky, p_risky=0.5)
    ib_h = make_ib_hypothesis(safe, risky)

    bayes = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=Greedy(),
    )
    ib = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[ib_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=None,
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
    bayes_h = make_bayesian_hypothesis(safe, risky, p_risky=0.5)
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[bayes_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=HypothesisThompsonSampling(),
    )
    agent.reset()
    probs = agent.get_probabilities()
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0)


def test_thompson_sampling_rejects_ku_multi_measure_hypothesis():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    ib_h = make_ib_hypothesis(safe, risky)
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[ib_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=HypothesisThompsonSampling(),
    )
    agent.reset()

    with pytest.raises(RuntimeError, match="exploration_strategy=None"):
        agent.get_probabilities()


def test_exploration_strategy_rejects_ku_multi_measure_hypothesis():
    _wm, safe, risky = make_trap_bandit_hypotheses(num_grid=3, p_cat=0.01)
    ib_h = make_ib_hypothesis(safe, risky)
    agent = InfraBayesianAgent(
        num_actions=2,
        hypotheses=[ib_h],
        prior=np.array([1.0]),
        reward_function=REWARD_FUNCTION,
        exploration_strategy=Greedy(),
    )
    agent.reset()

    with pytest.raises(RuntimeError, match="exploration_strategy=None"):
        agent.get_probabilities()
