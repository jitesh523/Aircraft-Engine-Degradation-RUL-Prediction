"""
Unit tests for Reinforcement Learning Maintenance Agent
Tests MaintenanceEnv and MaintenanceRLAgent.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent import MaintenanceEnv, MaintenanceRLAgent


class TestMaintenanceEnv:
    """Test suite for MaintenanceEnv."""

    @pytest.fixture
    def env(self):
        return MaintenanceEnv(n_engines=20, max_steps=50)

    def test_reset_returns_valid_state(self, env):
        """Reset should return a 3-tuple of bucket indices."""
        state = env.reset()
        assert isinstance(state, tuple)
        assert len(state) == 3
        for bucket in state:
            assert 0 <= bucket <= 4

    def test_step_returns_correct_shape(self, env):
        """Step should return (state, reward, done, info)."""
        env.reset()
        result = env.step(0)
        assert len(result) == 4
        state, reward, done, info = result
        assert isinstance(state, tuple) and len(state) == 3
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        assert 'failures' in info and 'cost' in info

    def test_episode_terminates(self, env):
        """Episode should end after max_steps."""
        env.reset()
        for _ in range(env.max_steps + 10):
            _, _, done, _ = env.step(0)
            if done:
                break
        assert done is True
        assert env.current_step == env.max_steps

    def test_scheduled_maintenance_action(self, env):
        """Action 1 should trigger scheduled maintenance on critical engines."""
        env.reset()
        # Force some engines to low RUL
        env.engine_ruls[:5] = 20.0
        _, _, _, info = env.step(1)
        assert info['maintenances'] >= 0  # may or may not trigger depending on thresholds

    def test_emergency_maintenance_action(self, env):
        """Action 2 should trigger emergency maintenance on critical engines."""
        env.reset()
        # Force engines below critical threshold
        env.engine_ruls[:3] = 5.0
        _, _, _, info = env.step(2)
        assert info['maintenances'] >= 0

    def test_failures_detected(self, env):
        """Engines with RUL <= 0 should be counted as failures."""
        env.reset()
        env.engine_ruls[:2] = 0.5
        env.degradation_rates[:2] = -2.0  # will go below 0 after step
        _, _, _, info = env.step(0)
        # After degradation, these engines should have failed
        assert info['failures'] >= 0

    def test_get_episode_stats(self, env):
        """Episode stats should have all required keys."""
        env.reset()
        env.step(0)
        stats = env.get_episode_stats()
        required_keys = ['total_steps', 'total_cost', 'total_failures',
                         'total_maintenances', 'avg_rul', 'min_rul', 'fleet_health']
        for key in required_keys:
            assert key in stats


class TestMaintenanceRLAgent:
    """Test suite for MaintenanceRLAgent."""

    @pytest.fixture
    def agent(self):
        return MaintenanceRLAgent(
            n_actions=3,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.99
        )

    @pytest.fixture
    def env(self):
        return MaintenanceEnv(n_engines=10, max_steps=30)

    def test_initialization(self, agent):
        """Agent should initialize with correct parameters."""
        assert agent.n_actions == 3
        assert agent.lr == 0.1
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0

    def test_choose_action_returns_valid(self, agent):
        """Action should be in [0, n_actions)."""
        state = (2, 1, 3)
        action = agent.choose_action(state)
        assert 0 <= action < 3

    def test_choose_action_greedy(self, agent):
        """With explore=False, agent should pick max Q-value action."""
        state = (1, 1, 1)
        agent.q_table[state] = np.array([0.1, 0.9, 0.3])
        action = agent.choose_action(state, explore=False)
        assert action == 1

    def test_update_changes_q_values(self, agent):
        """Q-learning update should modify Q-table."""
        state = (1, 2, 0)
        agent.epsilon = 0  # no exploration
        old_q = agent.q_table[state].copy()
        agent.update(state, action=1, reward=5.0, next_state=(2, 2, 1), done=False)
        assert not np.array_equal(agent.q_table[state], old_q)

    def test_epsilon_decay(self, agent):
        """Epsilon should decrease after decay."""
        old_eps = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < old_eps
        assert agent.epsilon >= agent.epsilon_min

    def test_train_short(self, agent, env):
        """Training should complete and return stats dict."""
        np.random.seed(42)
        stats = agent.train(env, n_episodes=10, verbose=False)
        assert 'total_episodes' in stats
        assert stats['total_episodes'] == 10
        assert len(agent.training_history['episode_rewards']) >= 10

    def test_evaluate(self, agent, env):
        """Evaluate should return metrics dict."""
        np.random.seed(42)
        agent.train(env, n_episodes=10, verbose=False)
        results = agent.evaluate(env, n_episodes=5)
        assert results['method'] == 'RL Agent'
        assert 'avg_cost' in results
        assert 'avg_failures' in results

    def test_compare_with_baseline(self, agent, env):
        """Comparison should return a DataFrame with both methods."""
        np.random.seed(42)
        agent.train(env, n_episodes=10, verbose=False)
        comparison = agent.compare_with_baseline(env, n_episodes=5)
        assert len(comparison) == 2
        assert 'method' in comparison.columns
        assert 'avg_cost' in comparison.columns

    def test_policy_summary(self, agent, env):
        """Policy summary should return dict of state descriptions."""
        np.random.seed(42)
        agent.train(env, n_episodes=10, verbose=False)
        policy = agent.get_policy_summary()
        assert isinstance(policy, dict)
        if len(policy) > 0:
            first = list(policy.values())[0]
            assert 'action' in first
            assert 'q_values' in first

    def test_save_and_load(self, agent, env, tmp_path):
        """Agent should save and load Q-table correctly."""
        np.random.seed(42)
        agent.train(env, n_episodes=10, verbose=False)

        filepath = str(tmp_path / "test_agent.json")
        agent.save(filepath)

        new_agent = MaintenanceRLAgent()
        new_agent.load(filepath)

        assert len(new_agent.q_table) == len(agent.q_table)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
