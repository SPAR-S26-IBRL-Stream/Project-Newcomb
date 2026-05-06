"""Tests for InfraBayesianAgent._solve_game (the 2-action maximin closed form)."""
import numpy as np
import pytest

from ibrl.agents.infrabayesian import InfraBayesianAgent
from ibrl.infrabayesian import GaussianBelief, NewcombLikeBelief


def _agent_2action():
    return InfraBayesianAgent(num_actions=2, seed=0,
                              beliefs=[GaussianBelief(2)])


class TestPureSaddle:
    def test_action_0_strictly_dominates(self):
        # V[0,0] best on diagonal, V[1,1] worse — predictor matters but
        # action 0 is the safer pure strategy. Maximin should pick p₀=1.
        agent = _agent_2action()
        V = np.array([[1.0, -1.0],
                      [-1.0, 0.5]])
        policy = agent._solve_game(V)
        assert policy[0] == pytest.approx(1.0)
        assert policy[1] == pytest.approx(0.0)

    def test_action_1_strictly_dominates(self):
        agent = _agent_2action()
        V = np.array([[0.5, -1.0],
                      [-1.0, 1.0]])
        policy = agent._solve_game(V)
        assert policy[1] == pytest.approx(1.0)


class TestMixedSaddle:
    def test_classic_2x2_zero_sum_mixed(self):
        # Anti-coordination matrix: agent should mix.
        # V = [[ 0, 1],
        #      [ 1, 0]]   ⇒ symmetric mixed at p₀ = 1/2.
        agent = _agent_2action()
        V = np.array([[0.0, 1.0],
                      [1.0, 0.0]])
        policy = agent._solve_game(V)
        # Pure: V[0,0]=0, V[1,1]=0 → both yield 0.
        # Mixed denom = 0 + 0 - 1 - 1 = -2 < 0; p₀_mixed = (1+1-0)/(1+1-0)/2 = 0.5;
        # value = (0 - 1)/(-2) = 0.5. Mixed wins.
        assert policy[0] == pytest.approx(0.5, abs=1e-9)


class TestThreeActionRejected:
    def test_solve_game_raises_for_3x3(self):
        agent = _agent_2action()
        V = np.zeros((3, 3))
        with pytest.raises(NotImplementedError):
            agent._solve_game(V)


class TestConstructionGuard:
    def test_3_actions_with_2d_belief_raises(self):
        # NewcombLikeBelief returns a 2-D model
        with pytest.raises(NotImplementedError, match="2-action closed-form"):
            InfraBayesianAgent(num_actions=3, seed=0,
                               beliefs=[NewcombLikeBelief(num_actions=3)])

    def test_2_actions_with_2d_belief_ok(self):
        # 2-action with NewcombLikeBelief should construct fine
        InfraBayesianAgent(num_actions=2, seed=0,
                           beliefs=[NewcombLikeBelief(num_actions=2)])

    def test_1d_beliefs_unaffected(self):
        # 1-D belief at any num_actions should construct without issue
        InfraBayesianAgent(num_actions=5, seed=0,
                           beliefs=[GaussianBelief(5)])
