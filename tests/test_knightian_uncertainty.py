"""Tests for Phase 4: Knightian Uncertainty (Definition 11 update rule).

These tests verify the KU update logic for belief-based infradistributions.
See experiments/alaro/docs/20260324_coinlearning.md §7.4-7.5 for the math.

Hand-computed test values use this setup:
    2 arms, 2 BernoulliBelief measures with different priors for arm 0:
        Belief A: Beta(2, 4) for arm 0 -> p_A = 2/6 = 1/3
        Belief B: Beta(4, 2) for arm 0 -> p_B = 4/6 = 2/3
    Both start with lambda=1, b=0.
    Observation: arm 0, reward 1.
"""
import numpy as np
import pytest

from ibrl.outcome import Outcome
from ibrl.infrabayesian.beliefs import BernoulliBelief, NewcombLikeBelief
from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.agents.infrabayesian import InfraBayesianAgent


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_bernoulli_belief(num_actions, arm_alpha_beta=None):
    """Create a BernoulliBelief with custom per-arm priors.

    arm_alpha_beta: dict mapping arm_index -> (alpha, beta).
    Unspecified arms get the default uniform Beta(1, 1).
    """
    b = BernoulliBelief(num_actions=num_actions)
    if arm_alpha_beta:
        for arm, (alpha, beta) in arm_alpha_beta.items():
            b.alpha[arm] = alpha
            b.beta[arm] = beta
    return b


def _make_two_measure_setup(g=1.0):
    """Standard test setup: 2 measures with different priors for arm 0.

    Belief A: Beta(2,4) for arm 0 -> p_A = 1/3
    Belief B: Beta(4,2) for arm 0 -> p_B = 2/3
    Both arms have Beta(1,1) for arm 1 -> p = 0.5
    """
    belief_a = _make_bernoulli_belief(2, {0: (2, 4)})
    belief_b = _make_bernoulli_belief(2, {0: (4, 2)})
    measures = [AMeasure(belief_a), AMeasure(belief_b)]
    return Infradistribution(measures, g=g)


# ── observation_probability ────────────────────────────────────────────────

class TestObservationProbability:
    def test_bernoulli_reward_1(self):
        """P(reward=1 | arm=0) = alpha / (alpha + beta)."""
        b = _make_bernoulli_belief(2, {0: (3, 7)})  # p = 0.3
        prob = b.observation_probability(action=0, outcome=Outcome(reward=1.0))
        assert prob == pytest.approx(0.3)

    def test_bernoulli_reward_0(self):
        """P(reward=0 | arm=0) = beta / (alpha + beta)."""
        b = _make_bernoulli_belief(2, {0: (3, 7)})  # p = 0.3
        prob = b.observation_probability(action=0, outcome=Outcome(reward=0.0))
        assert prob == pytest.approx(0.7)

    def test_bernoulli_uniform_prior(self):
        """Uniform Beta(1,1) -> P(reward=1) = P(reward=0) = 0.5."""
        b = BernoulliBelief(num_actions=2)
        assert b.observation_probability(0, Outcome(reward=1.0)) == pytest.approx(0.5)
        assert b.observation_probability(0, Outcome(reward=0.0)) == pytest.approx(0.5)

    def test_bernoulli_invalid_reward_raises(self):
        b = BernoulliBelief(num_actions=2)
        with pytest.raises(ValueError, match="reward in \\[0, 1\\]"):
            b.observation_probability(0, Outcome(reward=1.5))

    def test_newcomb_unobserved_cell_returns_1(self):
        """Unobserved cell: any outcome is 'expected', probability = 1."""
        b = NewcombLikeBelief(num_actions=2)
        prob = b.observation_probability(0, Outcome(reward=0.7, env_action=0))
        assert prob == 1.0

    def test_newcomb_observed_cell_match(self):
        """Observed cell matching reward -> probability = 1."""
        b = NewcombLikeBelief(num_actions=2)
        b.update(action=0, outcome=Outcome(reward=1.0, env_action=0))
        prob = b.observation_probability(0, Outcome(reward=1.0, env_action=0))
        assert prob == 1.0

    def test_newcomb_observed_cell_mismatch(self):
        """Observed cell not matching reward -> probability = 0."""
        b = NewcombLikeBelief(num_actions=2)
        b.update(action=0, outcome=Outcome(reward=1.0, env_action=0))
        prob = b.observation_probability(0, Outcome(reward=0.0, env_action=0))
        assert prob == 0.0


# ── KU update: non-KU unchanged (test 1 from §7.5.6) ──────────────────────

class TestNonKUUnchanged:
    """Single-measure infradistribution with KU update path should produce
    identical results to the current non-KU behavior: lambda=1, b=0 throughout."""

    def test_single_measure_scale_stays_one(self):
        belief = BernoulliBelief(num_actions=2)
        infradist = Infradistribution([AMeasure(belief)], g=1.0)

        for _ in range(10):
            infradist.update(action=0, outcome=Outcome(reward=1.0))

        m = infradist.measures[0]
        assert m.log_scale == pytest.approx(0.0, abs=1e-12)
        assert m.offset == pytest.approx(0.0, abs=1e-12)

    def test_single_measure_model_matches_direct_belief(self):
        belief_direct = BernoulliBelief(num_actions=2)
        belief_infra = BernoulliBelief(num_actions=2)
        infradist = Infradistribution([AMeasure(belief_infra)], g=1.0)

        rng = np.random.default_rng(42)
        for _ in range(20):
            action = rng.integers(2)
            reward = float(rng.random() < 0.7) if action == 0 else float(rng.random() < 0.3)
            outcome = Outcome(reward=reward)

            belief_direct.update(action=action, outcome=outcome)
            infradist.update(action=action, outcome=outcome)

            np.testing.assert_allclose(
                infradist.expected_reward_model(),
                belief_direct.expected_reward_model(),
                atol=1e-12,
            )


# ── KU update: offset and scale (tests 2-3 from §7.5.6) ──────────────────

class TestKUUpdateMath:
    """Verify KU update produces correct lambda and b values.

    Hand-computed for g=1, two measures, observation (arm 0, reward 1):

        Belief A: p_A = 1/3, Belief B: p_B = 2/3

        cfval_A = 1 * 1 * 2/3 + 0 = 2/3
        cfval_B = 1 * 1 * 1/3 + 0 = 1/3
        global_cfval = 1/3

        full_A = 1*1/3 + 1*1*2/3 + 0 = 1
        full_B = 1*2/3 + 1*1*1/3 + 0 = 1
        global_full = 1

        P^g_H(L) = 1 - 1/3 = 2/3

        lambda_A_new = 1 * (1/3) / (2/3) = 1/2
        lambda_B_new = 1 * (2/3) / (2/3) = 1

        b_A_new = (2/3 - 1/3) / (2/3) = 1/2
        b_B_new = (1/3 - 1/3) / (2/3) = 0
    """

    def test_scale_after_update(self):
        infradist = _make_two_measure_setup(g=1.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        lambda_a = np.exp(infradist.measures[0].log_scale)
        lambda_b = np.exp(infradist.measures[1].log_scale)
        assert lambda_a == pytest.approx(0.5)
        assert lambda_b == pytest.approx(1.0)

    def test_offset_after_update(self):
        infradist = _make_two_measure_setup(g=1.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        assert infradist.measures[0].offset == pytest.approx(0.5)
        assert infradist.measures[1].offset == pytest.approx(0.0)

    def test_cohomogeneity_preserved_with_g1(self):
        """With g=1, lambda + b should equal 1 for all measures after update."""
        infradist = _make_two_measure_setup(g=1.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        for m in infradist.measures:
            lam = np.exp(m.log_scale)
            assert lam + m.offset == pytest.approx(1.0)

    def test_beliefs_updated_bayesian(self):
        """After KU update, each belief should have standard Bayesian posterior."""
        infradist = _make_two_measure_setup(g=1.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        # Belief A: was Beta(2,4) for arm 0 -> now Beta(3,4), p = 3/7
        model_a = infradist.measures[0].belief.expected_reward_model()
        assert model_a[0] == pytest.approx(3.0 / 7.0)
        assert model_a[1] == pytest.approx(0.5)  # arm 1 unchanged

        # Belief B: was Beta(4,2) for arm 0 -> now Beta(5,2), p = 5/7
        model_b = infradist.measures[1].belief.expected_reward_model()
        assert model_b[0] == pytest.approx(5.0 / 7.0)
        assert model_b[1] == pytest.approx(0.5)

    def test_expected_reward_model_is_elementwise_min(self):
        """Infradistribution model = min over a-measure models."""
        infradist = _make_two_measure_setup(g=1.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        # Measure A model: lambda=1/2, belief=[3/7, 0.5], offset=1/2
        #   arm 0: 0.5 * 3/7 + 0.5 = 3/14 + 7/14 = 10/14 = 5/7
        #   arm 1: 0.5 * 0.5 + 0.5 = 0.75
        # Measure B model: lambda=1, belief=[5/7, 0.5], offset=0
        #   arm 0: 1 * 5/7 + 0 = 5/7
        #   arm 1: 1 * 0.5 + 0 = 0.5
        # Min: arm 0 = 5/7, arm 1 = 0.5

        model = infradist.expected_reward_model()
        assert model[0] == pytest.approx(5.0 / 7.0)
        assert model[1] == pytest.approx(0.5)

    def test_multi_step_update(self):
        """KU update over multiple steps doesn't crash and maintains invariants."""
        infradist = _make_two_measure_setup(g=1.0)
        rng = np.random.default_rng(99)

        for _ in range(20):
            action = rng.integers(2)
            reward = float(rng.random() < 0.6)
            infradist.update(action=action, outcome=Outcome(reward=reward))

            # Check invariants after each step
            for m in infradist.measures:
                lam = np.exp(m.log_scale)
                assert lam > 0, f"lambda must be > 0, got {lam}"
                assert m.offset >= -1e-12, f"offset must be >= 0, got {m.offset}"
                # g=1 cohomogeneity
                assert lam + m.offset == pytest.approx(1.0, abs=1e-10)


# ── KU update: g=0 degeneracy (test 4 from §7.5.6) ───────────────────────

class TestGZeroDegeneracy:
    """With g=0, offsets should remain 0 after update.

    Hand-computed:
        cfval_A = cfval_B = 0, global_cfval = 0
        full_A = 1/3, full_B = 2/3, global_full = 1/3
        P^g_H(L) = 1/3
        lambda_A_new = (1/3)/(1/3) = 1
        lambda_B_new = (2/3)/(1/3) = 2
        b_A_new = 0, b_B_new = 0
    """

    def test_offsets_stay_zero(self):
        infradist = _make_two_measure_setup(g=0.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        assert infradist.measures[0].offset == pytest.approx(0.0)
        assert infradist.measures[1].offset == pytest.approx(0.0)

    def test_scales_with_g_zero(self):
        infradist = _make_two_measure_setup(g=0.0)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        lambda_a = np.exp(infradist.measures[0].log_scale)
        lambda_b = np.exp(infradist.measures[1].log_scale)
        assert lambda_a == pytest.approx(1.0)
        assert lambda_b == pytest.approx(2.0)


# ── KU update: g=0.5 non-trivial (test 5 from §7.5.6) ────────────────────

class TestGNonTrivial:
    """With g=0.5, offsets should become non-zero.

    Hand-computed:
        cfval_A = 0.5 * 1 * 2/3 + 0 = 1/3
        cfval_B = 0.5 * 1 * 1/3 + 0 = 1/6
        global_cfval = 1/6

        full_A = 1/3 + 0.5*2/3 = 2/3
        full_B = 2/3 + 0.5*1/3 = 5/6
        global_full = 2/3

        P^g_H(L) = 2/3 - 1/6 = 1/2

        lambda_A_new = (1/3) / (1/2) = 2/3
        lambda_B_new = (2/3) / (1/2) = 4/3

        b_A_new = (1/3 - 1/6) / (1/2) = (1/6)/(1/2) = 1/3
        b_B_new = (1/6 - 1/6) / (1/2) = 0
    """

    def test_offsets_nonzero_with_g_half(self):
        infradist = _make_two_measure_setup(g=0.5)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        assert infradist.measures[0].offset == pytest.approx(1.0 / 3.0)
        assert infradist.measures[1].offset == pytest.approx(0.0)

    def test_scales_with_g_half(self):
        infradist = _make_two_measure_setup(g=0.5)
        infradist.update(action=0, outcome=Outcome(reward=1.0))

        lambda_a = np.exp(infradist.measures[0].log_scale)
        lambda_b = np.exp(infradist.measures[1].log_scale)
        assert lambda_a == pytest.approx(2.0 / 3.0)
        assert lambda_b == pytest.approx(4.0 / 3.0)


# ── Validation checks ─────────────────────────────────────────────────────

class TestValidation:
    def test_g_out_of_range_raises(self):
        belief = BernoulliBelief(num_actions=2)
        with pytest.raises(ValueError, match="g must be in"):
            Infradistribution([AMeasure(belief)], g=1.5)

    def test_empty_measures_raises(self):
        with pytest.raises(ValueError, match="at least one measure"):
            Infradistribution([], g=1.0)

    def test_agent_empty_beliefs_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            InfraBayesianAgent(num_actions=2, beliefs=[], epsilon=0.1)

    def test_utility_out_of_range_raises(self):
        """Utility mapping that produces values outside [0,1] should raise."""
        bad_utility = lambda r: r * 2.0  # doubles reward, can exceed 1
        agent = InfraBayesianAgent(
            num_actions=2,
            beliefs=[BernoulliBelief(num_actions=2)],
            epsilon=0.1,
            utility=bad_utility,
        )
        agent.reset()
        probs = agent.get_probabilities()
        with pytest.raises(ValueError, match="Utility mapping must produce"):
            agent.update(probs, 0, Outcome(reward=0.8))


# ── Agent integration with beliefs= kwarg ─────────────────────────────────

class TestAgentBeliefsKwarg:
    def test_single_belief_list_works(self):
        """beliefs=[single_belief] should work like the old belief= interface."""
        agent = InfraBayesianAgent(
            num_actions=2,
            beliefs=[BernoulliBelief(num_actions=2)],
            epsilon=0.1,
        )
        agent.reset()
        probs = agent.get_probabilities()
        assert probs.shape == (2,)
        assert probs.sum() == pytest.approx(1.0)

    def test_multiple_beliefs_creates_ku_infradist(self):
        """beliefs=[b1, b2, b3] should create infradist with 3 measures."""
        beliefs = [
            _make_bernoulli_belief(2, {0: (1, 3)}),
            _make_bernoulli_belief(2, {0: (2, 2)}),
            _make_bernoulli_belief(2, {0: (3, 1)}),
        ]
        agent = InfraBayesianAgent(
            num_actions=2, beliefs=beliefs, epsilon=0.1,
        )
        agent.reset()
        assert len(agent.infradist.measures) == 3

    def test_agent_ku_update_runs(self):
        """KU agent should complete update/get_probabilities cycle."""
        beliefs = [
            _make_bernoulli_belief(2, {0: (2, 4)}),
            _make_bernoulli_belief(2, {0: (4, 2)}),
        ]
        agent = InfraBayesianAgent(
            num_actions=2, beliefs=beliefs, epsilon=0.1,
        )
        agent.reset()

        for _ in range(10):
            probs = agent.get_probabilities()
            assert probs.shape == (2,)
            agent.update(probs, 0, Outcome(reward=1.0))
