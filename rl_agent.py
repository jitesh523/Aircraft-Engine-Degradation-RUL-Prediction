"""
Reinforcement Learning Agent for Autonomous Maintenance Optimization
Learns optimal maintenance scheduling policies to minimize costs and failures.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import os
import config
from utils import setup_logging

logger = setup_logging(__name__)


# ============================================================
# Maintenance Environment (Gym-like interface)
# ============================================================
class MaintenanceEnv:
    """
    Simulated maintenance environment for RL training.
    
    State: (current_rul_bucket, degradation_rate_bucket, days_since_last_maintenance)
    Actions: 0 = Do Nothing, 1 = Schedule Maintenance, 2 = Emergency Maintenance
    
    Reward: Negative cost (agent tries to maximize reward = minimize cost).
    """
    
    # Discretized state space
    RUL_BUCKETS = [0, 15, 30, 50, 80, 125]  # 5 buckets
    DEGRADATION_BUCKETS = [-3.0, -2.0, -1.0, -0.5, 0.0, 1.0]  # 5 buckets
    MAINTENANCE_AGE_BUCKETS = [0, 10, 25, 50, 100, 200]  # 5 buckets
    
    ACTION_NAMES = {0: "Do Nothing", 1: "Schedule Maintenance", 2: "Emergency Maintenance"}
    
    def __init__(self, 
                 n_engines: int = 100,
                 max_steps: int = 300,
                 cost_params: Dict = None):
        """
        Initialize the maintenance environment.
        
        Args:
            n_engines: Number of engines to simulate.
            max_steps: Maximum simulation steps (cycles).
            cost_params: Cost parameters dictionary.
        """
        self.n_engines = n_engines
        self.max_steps = max_steps
        self.costs = cost_params or config.COST_PARAMETERS
        
        # State tracking
        self.current_step = 0
        self.engine_ruls = None
        self.degradation_rates = None
        self.maintenance_ages = None
        self.total_cost = 0
        self.failures = 0
        self.maintenances = 0
        self.done = False
        
        logger.info(f"MaintenanceEnv initialized: {n_engines} engines, {max_steps} max steps")
    
    def reset(self) -> Tuple:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state tuple.
        """
        self.current_step = 0
        self.total_cost = 0
        self.failures = 0
        self.maintenances = 0
        self.done = False
        
        # Initialize engine RULs (random starting lifetimes)
        self.engine_ruls = np.random.uniform(50, 125, self.n_engines)
        
        # Degradation rates (cycles lost per step, negative = degrading)
        self.degradation_rates = np.random.uniform(-2.0, -0.3, self.n_engines)
        
        # Steps since last maintenance
        self.maintenance_ages = np.zeros(self.n_engines)
        
        return self._get_state()
    
    def _get_state(self) -> Tuple:
        """
        Get the aggregated fleet state (discretized).
        
        Returns:
            State tuple: (min_rul_bucket, avg_degradation_bucket, max_maintenance_age_bucket)
        """
        min_rul = np.min(self.engine_ruls)
        avg_deg = np.mean(self.degradation_rates)
        max_age = np.max(self.maintenance_ages)
        
        rul_bucket = np.digitize(min_rul, self.RUL_BUCKETS) - 1
        deg_bucket = np.digitize(avg_deg, self.DEGRADATION_BUCKETS) - 1
        age_bucket = np.digitize(max_age, self.MAINTENANCE_AGE_BUCKETS) - 1
        
        # Clamp to valid range
        rul_bucket = max(0, min(rul_bucket, len(self.RUL_BUCKETS) - 2))
        deg_bucket = max(0, min(deg_bucket, len(self.DEGRADATION_BUCKETS) - 2))
        age_bucket = max(0, min(age_bucket, len(self.MAINTENANCE_AGE_BUCKETS) - 2))
        
        return (rul_bucket, deg_bucket, age_bucket)
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Take an action in the environment.
        
        Args:
            action: 0=nothing, 1=scheduled maintenance, 2=emergency maintenance
            
        Returns:
            (next_state, reward, done, info)
        """
        reward = 0.0
        info = {'failures': 0, 'maintenances': 0, 'cost': 0.0}
        
        # --- Apply action ---
        if action == 1:  # Scheduled maintenance on most critical engines
            needs_maint = self.engine_ruls < config.MAINTENANCE_THRESHOLDS['warning']
            if needs_maint.any():
                n_maintained = min(5, needs_maint.sum())  # Maintain up to 5
                critical_indices = np.argsort(self.engine_ruls)[:n_maintained]
                
                for idx in critical_indices:
                    cost = self.costs['scheduled_maintenance']
                    reward -= cost / 100000.0  # Normalize: smaller penalty for scheduled
                    info['cost'] += cost
                    self.maintenances += 1
                    info['maintenances'] += 1
                    
                    # Bonus for proactive maintenance (preventing failure)
                    if self.engine_ruls[idx] < config.MAINTENANCE_THRESHOLDS['critical']:
                        reward += 2.0  # Saved a near-failure!
                    
                    # Restore engine
                    self.engine_ruls[idx] = np.random.uniform(100, 125)
                    self.degradation_rates[idx] = np.random.uniform(-1.5, -0.3)
                    self.maintenance_ages[idx] = 0
        
        elif action == 2:  # Emergency maintenance on all critical engines
            emergency_mask = self.engine_ruls < config.MAINTENANCE_THRESHOLDS['critical']
            if emergency_mask.any():
                for idx in np.where(emergency_mask)[0]:
                    cost = self.costs['unscheduled_maintenance']
                    reward -= cost / 50000.0  # Higher penalty for emergency
                    info['cost'] += cost
                    self.maintenances += 1
                    info['maintenances'] += 1
                    
                    # Restore engine
                    self.engine_ruls[idx] = np.random.uniform(100, 125)
                    self.degradation_rates[idx] = np.random.uniform(-1.5, -0.3)
                    self.maintenance_ages[idx] = 0
        
        # --- Simulate one timestep of degradation ---
        self.engine_ruls += self.degradation_rates
        self.maintenance_ages += 1
        
        # Add noise to degradation rates
        self.degradation_rates += np.random.normal(0, 0.05, self.n_engines)
        self.degradation_rates = np.clip(self.degradation_rates, -3.0, -0.1)
        
        # --- Check for failures (LARGE penalty) ---
        failed_mask = self.engine_ruls <= 0
        n_failures = failed_mask.sum()
        if n_failures > 0:
            failure_cost = n_failures * self.costs['unscheduled_maintenance'] * 2
            reward -= n_failures * 10.0  # Very large penalty per failure
            info['failures'] = n_failures
            info['cost'] += failure_cost
            self.failures += n_failures
            
            # Reset failed engines (replacement)
            for idx in np.where(failed_mask)[0]:
                self.engine_ruls[idx] = np.random.uniform(100, 125)
                self.degradation_rates[idx] = np.random.uniform(-1.5, -0.3)
                self.maintenance_ages[idx] = 0
        
        # Small positive reward for keeping fleet running
        healthy_ratio = np.mean(self.engine_ruls > config.MAINTENANCE_THRESHOLDS['critical'])
        reward += healthy_ratio * 0.1
        
        # Update step counter
        self.current_step += 1
        self.total_cost += info['cost']
        
        if self.current_step >= self.max_steps:
            self.done = True
        
        return self._get_state(), reward, self.done, info
    
    def get_episode_stats(self) -> Dict:
        """Get statistics for the current episode."""
        return {
            'total_steps': self.current_step,
            'total_cost': self.total_cost,
            'total_failures': self.failures,
            'total_maintenances': self.maintenances,
            'avg_rul': float(np.mean(self.engine_ruls)),
            'min_rul': float(np.min(self.engine_ruls)),
            'fleet_health': float(np.mean(self.engine_ruls > config.MAINTENANCE_THRESHOLDS['critical']) * 100)
        }


# ============================================================
# Q-Learning Agent
# ============================================================
class MaintenanceRLAgent:
    """
    Q-Learning agent for maintenance optimization.
    
    Learns a policy that maps fleet states to maintenance actions
    to minimize long-term costs while preventing failures.
    """
    
    def __init__(self, 
                 n_actions: int = 3,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of possible actions.
            learning_rate: Learning rate (alpha).
            discount_factor: Discount factor (gamma).
            epsilon: Initial exploration rate.
            epsilon_min: Minimum exploration rate.
            epsilon_decay: Exploration decay per episode.
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_costs': [],
            'episode_failures': [],
            'epsilon_values': []
        }
        
        logger.info(f"RL Agent initialized: lr={learning_rate}, gamma={discount_factor}, "
                    f"epsilon={epsilon}")
    
    def choose_action(self, state: Tuple, explore: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state tuple.
            explore: Whether to use exploration (False for evaluation).
            
        Returns:
            Selected action index.
        """
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = self.q_table[state]
        return int(np.argmax(q_values))
    
    def update(self, state: Tuple, action: int, reward: float, 
               next_state: Tuple, done: bool):
        """
        Update Q-value using the Bellman equation.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        current_q = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        
        # Q-learning update
        self.q_table[state][action] = current_q + self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env: MaintenanceEnv, n_episodes: int = 500,
              verbose: bool = True) -> Dict:
        """
        Train the agent on the maintenance environment.
        
        Args:
            env: MaintenanceEnv instance.
            n_episodes: Number of training episodes.
            verbose: Whether to print progress.
            
        Returns:
            Training statistics.
        """
        logger.info(f"Starting RL training: {n_episodes} episodes")
        
        best_reward = float('-inf')
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
            
            # Record episode stats
            stats = env.get_episode_stats()
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['episode_costs'].append(stats['total_cost'])
            self.training_history['episode_failures'].append(stats['total_failures'])
            self.training_history['epsilon_values'].append(self.epsilon)
            
            self.decay_epsilon()
            
            if total_reward > best_reward:
                best_reward = total_reward
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-50:])
                avg_cost = np.mean(self.training_history['episode_costs'][-50:])
                avg_failures = np.mean(self.training_history['episode_failures'][-50:])
                logger.info(
                    f"Episode {episode+1}/{n_episodes} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Cost: ${avg_cost:,.0f} | "
                    f"Avg Failures: {avg_failures:.1f} | "
                    f"ε: {self.epsilon:.3f}"
                )
        
        logger.info(f"Training complete. Best reward: {best_reward:.2f}")
        
        return {
            'total_episodes': n_episodes,
            'best_reward': best_reward,
            'final_epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'avg_reward_last_50': float(np.mean(self.training_history['episode_rewards'][-50:])),
            'avg_cost_last_50': float(np.mean(self.training_history['episode_costs'][-50:])),
            'avg_failures_last_50': float(np.mean(self.training_history['episode_failures'][-50:]))
        }
    
    def evaluate(self, env: MaintenanceEnv, n_episodes: int = 50) -> Dict:
        """
        Evaluate the trained agent without exploration.
        
        Args:
            env: MaintenanceEnv instance.
            n_episodes: Number of evaluation episodes.
            
        Returns:
            Evaluation statistics.
        """
        logger.info(f"Evaluating RL agent over {n_episodes} episodes...")
        
        rewards, costs, failures = [], [], []
        
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = self.choose_action(state, explore=False)
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
            
            stats = env.get_episode_stats()
            rewards.append(total_reward)
            costs.append(stats['total_cost'])
            failures.append(stats['total_failures'])
        
        results = {
            'method': 'RL Agent',
            'avg_reward': float(np.mean(rewards)),
            'avg_cost': float(np.mean(costs)),
            'avg_failures': float(np.mean(failures)),
            'std_cost': float(np.std(costs)),
            'min_cost': float(np.min(costs)),
            'max_cost': float(np.max(costs))
        }
        
        logger.info(f"RL Agent: Avg Cost=${results['avg_cost']:,.0f}, "
                    f"Avg Failures={results['avg_failures']:.1f}")
        
        return results
    
    def evaluate_baseline(self, env: MaintenanceEnv, 
                          threshold: int = 30,
                          n_episodes: int = 50) -> Dict:
        """
        Evaluate a simple threshold-based baseline for comparison.
        
        Args:
            env: MaintenanceEnv instance.
            threshold: RUL threshold for triggering maintenance.
            n_episodes: Number of episodes.
            
        Returns:
            Baseline statistics.
        """
        logger.info(f"Evaluating threshold baseline (threshold={threshold})...")
        
        rewards, costs, failures = [], [], []
        
        for _ in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                # Simple rule: if min RUL is in critical bucket, do emergency
                # If in warning bucket, schedule maintenance
                rul_bucket = state[0]
                if rul_bucket <= 1:  # Critical
                    action = 2
                elif rul_bucket <= 2:  # Warning
                    action = 1
                else:
                    action = 0
                
                next_state, reward, done, info = env.step(action)
                state = next_state
                total_reward += reward
            
            stats = env.get_episode_stats()
            rewards.append(total_reward)
            costs.append(stats['total_cost'])
            failures.append(stats['total_failures'])
        
        results = {
            'method': f'Threshold Baseline (t={threshold})',
            'avg_reward': float(np.mean(rewards)),
            'avg_cost': float(np.mean(costs)),
            'avg_failures': float(np.mean(failures)),
            'std_cost': float(np.std(costs)),
            'min_cost': float(np.min(costs)),
            'max_cost': float(np.max(costs))
        }
        
        logger.info(f"Baseline: Avg Cost=${results['avg_cost']:,.0f}, "
                    f"Avg Failures={results['avg_failures']:.1f}")
        
        return results
    
    def compare_with_baseline(self, env: MaintenanceEnv, 
                               n_episodes: int = 50) -> pd.DataFrame:
        """
        Compare RL agent performance with threshold baseline.
        
        Args:
            env: MaintenanceEnv instance.
            n_episodes: Number of evaluation episodes.
            
        Returns:
            Comparison DataFrame.
        """
        rl_results = self.evaluate(env, n_episodes)
        baseline_results = self.evaluate_baseline(env, n_episodes=n_episodes)
        
        comparison = pd.DataFrame([baseline_results, rl_results])
        
        # Calculate improvement
        cost_improvement = (
            (baseline_results['avg_cost'] - rl_results['avg_cost']) 
            / baseline_results['avg_cost'] * 100
        ) if baseline_results['avg_cost'] > 0 else 0
        
        failure_improvement = (
            (baseline_results['avg_failures'] - rl_results['avg_failures'])
            / max(baseline_results['avg_failures'], 1) * 100
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RL vs Baseline Comparison")
        logger.info(f"{'='*60}")
        logger.info(f"Cost Improvement: {cost_improvement:.1f}%")
        logger.info(f"Failure Reduction: {failure_improvement:.1f}%")
        
        return comparison
    
    def get_policy_summary(self) -> Dict:
        """
        Get a human-readable summary of the learned policy.
        
        Returns:
            Dictionary mapping state descriptions to recommended actions.
        """
        rul_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        deg_labels = ['Rapid', 'Fast', 'Moderate', 'Slow', 'Stable']
        age_labels = ['Recent', 'Short', 'Medium', 'Long', 'Very Long']
        
        policy = {}
        for state, q_values in self.q_table.items():
            rul_b, deg_b, age_b = state
            
            rul_label = rul_labels[min(rul_b, len(rul_labels)-1)]
            deg_label = deg_labels[min(deg_b, len(deg_labels)-1)]
            age_label = age_labels[min(age_b, len(age_labels)-1)]
            
            best_action = int(np.argmax(q_values))
            action_name = MaintenanceEnv.ACTION_NAMES[best_action]
            
            state_desc = f"RUL={rul_label}, Degradation={deg_label}, Age={age_label}"
            policy[state_desc] = {
                'action': action_name,
                'q_values': {MaintenanceEnv.ACTION_NAMES[i]: round(v, 3) 
                            for i, v in enumerate(q_values)},
                'confidence': float(np.max(q_values) - np.mean(q_values))
            }
        
        return policy
    
    def save(self, filepath: str):
        """Save the trained agent to disk."""
        save_data = {
            'q_table': {str(k): v.tolist() for k, v in self.q_table.items()},
            'epsilon': self.epsilon,
            'training_history': self.training_history,
            'lr': self.lr,
            'gamma': self.gamma
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load a trained agent from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions))
        for k, v in data['q_table'].items():
            state = eval(k)  # Convert string tuple back
            self.q_table[state] = np.array(v)
        
        self.epsilon = data.get('epsilon', self.epsilon_min)
        self.training_history = data.get('training_history', self.training_history)
        
        logger.info(f"Agent loaded from {filepath} ({len(self.q_table)} states)")


# ============================================================
# Standalone training & evaluation
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Training RL Maintenance Agent")
    print("=" * 60)
    
    # Create environment
    env = MaintenanceEnv(n_engines=50, max_steps=200)
    
    # Create agent
    agent = MaintenanceRLAgent(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995
    )
    
    # Train
    print("\n--- Training ---")
    train_stats = agent.train(env, n_episodes=500, verbose=True)
    print(f"\nTraining stats: {json.dumps(train_stats, indent=2)}")
    
    # Compare with baseline
    print("\n--- Evaluation ---")
    comparison = agent.compare_with_baseline(env, n_episodes=50)
    print("\nComparison:")
    print(comparison.to_string(index=False))
    
    # Show learned policy
    print("\n--- Learned Policy ---")
    policy = agent.get_policy_summary()
    for state_desc, info in list(policy.items())[:10]:
        print(f"  {state_desc} → {info['action']} (confidence: {info['confidence']:.2f})")
    
    # Save agent
    save_path = os.path.join(config.MODELS_DIR, 'rl_agent.json')
    agent.save(save_path)
    print(f"\nAgent saved to: {save_path}")
