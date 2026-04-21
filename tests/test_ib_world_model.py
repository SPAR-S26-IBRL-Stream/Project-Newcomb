"""Phase 1 behavioral anchors for the WorldModel-based infrabayesian refactor."""
import numpy as np
import pytest

from ibrl.infrabayesian.a_measure import AMeasure
from ibrl.infrabayesian.infradistribution import Infradistribution
from ibrl.infrabayesian.world_model import MultiBernoulliWorldModel
from ibrl.outcome import Outcome

NUM_ARMS = 1
ARM = 0

REWARD = np.array([0., 1.])  # outcome 0 → reward 0, outcome 1 → reward 1


def make_dist(num_hypotheses: int = 5) -> Infradistribution:
    """Uniform Bayesian prior over a linspace grid of Bernoulli hypotheses."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    hypotheses = [
        Infradistribution(
            [AMeasure(wm.make_params([np.array([1 - p, p])]))],
            world_model=wm,
        )
        for p in np.linspace(0., 1., num_hypotheses)
    ]
    return Infradistribution.mix(hypotheses, np.ones(num_hypotheses) / num_hypotheses)


def obs(reward: float) -> Outcome:
    return Outcome(reward=reward)


# ── Normalization conditions ────────────────────────────────────────────────

def test_e_h_zero_is_zero():
    """E_H([0, 0]) == 0 at initialization."""
    dist = make_dist()
    assert abs(dist.evaluate_action(np.zeros(2), action=ARM)) < 1e-9


def test_e_h_zero_stays_zero_after_updates():
    """E_H([0, 0]) == 0 is preserved after updates."""
    dist = make_dist()
    rf = np.tile(REWARD, (NUM_ARMS, 1))
    for r in [1., 0., 1., 1., 0.]:
        dist.update(rf, obs(r), action=ARM)
    assert abs(dist.evaluate_action(np.zeros(2), action=ARM)) < 1e-9


def test_evaluate_action_in_unit_interval():
    dist = make_dist()
    v = dist.evaluate_action(REWARD, action=ARM)
    assert 0. <= v <= 1.


# ── Learning direction ──────────────────────────────────────────────────────

def test_update_increases_evaluate_action_after_rewards():
    dist = make_dist()
    rf = np.tile(REWARD, (NUM_ARMS, 1))
    ev_before = dist.evaluate_action(REWARD, action=ARM)
    for _ in range(20):
        dist.update(rf, obs(1.), action=ARM)
    assert dist.evaluate_action(REWARD, action=ARM) > ev_before


def test_update_decreases_evaluate_action_after_no_rewards():
    dist = make_dist()
    rf = np.tile(REWARD, (NUM_ARMS, 1))
    for _ in range(5):
        dist.update(rf, obs(1.), action=ARM)
    ev_before = dist.evaluate_action(REWARD, action=ARM)
    for _ in range(20):
        dist.update(rf, obs(0.), action=ARM)
    assert dist.evaluate_action(REWARD, action=ARM) < ev_before


# ── Scale and offset invariants ─────────────────────────────────────────────

def test_scale_and_offset_nonnegative():
    dist = make_dist()
    rf = np.tile(REWARD, (NUM_ARMS, 1))
    for r in [1., 0., 1., 1., 0., 0., 1.]:
        dist.update(rf, obs(r), action=ARM)
    for m in dist.measures:
        assert m.scale >= 0.
        assert m.offset >= 0.


def test_pure_measure_evaluate_action():
    """Single-component AMeasure: evaluate_action returns the hypothesis mean."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    dist = Infradistribution(
        [AMeasure(wm.make_params([np.array([0.3, 0.7])]))],
        world_model=wm,
    )
    assert abs(dist.evaluate_action(REWARD, action=ARM) - 0.7) < 1e-6


# ── Mix variants ────────────────────────────────────────────────────────────

def test_mix_bayesian_average():
    """Classical Bayesian mix of two hypotheses gives their weighted average."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    d_low  = Infradistribution([AMeasure(wm.make_params([np.array([0.8, 0.2])]))], world_model=wm)
    d_high = Infradistribution([AMeasure(wm.make_params([np.array([0.2, 0.8])]))], world_model=wm)
    dist = Infradistribution.mix([d_low, d_high], np.array([0.5, 0.5]))
    # Both arms equal weight → average expected reward = 0.5 * 0.2 + 0.5 * 0.8 = 0.5
    assert abs(dist.evaluate_action(REWARD, action=ARM) - 0.5) < 1e-6


def test_mixku_pessimistic():
    """KU mixture takes the pessimistic (min) expected reward."""
    wm = MultiBernoulliWorldModel(num_arms=NUM_ARMS)
    d_low  = Infradistribution([AMeasure(wm.make_params([np.array([0.8, 0.2])]))], world_model=wm)
    d_high = Infradistribution([AMeasure(wm.make_params([np.array([0.2, 0.8])]))], world_model=wm)
    dist = Infradistribution.mixKU([d_low, d_high])
    # Pessimistic: min(0.2, 0.8) = 0.2
    assert abs(dist.evaluate_action(REWARD, action=ARM) - 0.2) < 1e-6


# ── DiscreteBayesianAgent equivalence ──────────────────────────────────────

def test_bernoulli_grid_equivalent_to_discrete_bayesian():
    """
    An IB agent built with a uniform grid of Bernoulli hypotheses should choose
    the same arm as DiscreteBayesianAgent at every step. The raw expected values
    differ by a global scale factor (from the IB normalization), but argmax is
    preserved because the scale is the same across all arms.
    """
    from ibrl.agents.discrete_bayesian import DiscreteBayesianAgent
    from ibrl.agents.infrabayesian import InfraBayesianAgent

    n = 10
    num_hypotheses = 5

    wm = MultiBernoulliWorldModel(num_arms=n)
    grid = [np.array([1 - p, p]) for p in np.linspace(0., 1., num_hypotheses)]
    params = wm.make_params([grid] * n)
    hypotheses = [Infradistribution([AMeasure(params)], world_model=wm)]

    db = DiscreteBayesianAgent(num_actions=n, num_hypotheses=num_hypotheses,
                               epsilon=0.0, seed=0)
    ib = InfraBayesianAgent(num_actions=n, hypotheses=hypotheses,
                            prior=np.array([1.0]), epsilon=0.0, seed=0)
    db.reset()
    ib.reset()

    rng = np.random.default_rng(42)
    for step in range(40):
        db_policy = db.get_probabilities()
        ib_policy = ib.get_probabilities()
        
        # assert arm ordering is the same for both agents
        assert np.array_equal(np.argsort(db_policy), np.argsort(ib_policy)), (
            f"Step {step}: DB={db_policy}, IB={ib_policy}"
        )

        action = int(rng.integers(n))
        # hardcode first arm is the best
        reward = float(rng.random() < (0.7 if action == 0 else 0.3))
        outcome = Outcome(reward=reward)

        db.update(db_policy, action, outcome)
        ib.update(ib_policy, action, outcome)
