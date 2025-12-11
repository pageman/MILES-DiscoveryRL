#!/usr/bin/env python3
"""
Drug Optimization RL Training Script
====================================

Trains an RL agent to optimize drug selection balancing:
- Efficacy (target activity)
- Safety (low cytotoxicity)
- Selectivity (low promiscuity)

This demonstrates a simple Q-learning approach that could be scaled
using the MILES framework for enterprise deployment.

Connection to MILES:
- This toy example uses tabular Q-learning
- MILES would enable scaling to:
  * Large compound libraries (millions)
  * Complex molecular representations
  * Multi-target optimization
  * Distributed training across GPUs/nodes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple
import pickle
from drug_rl_environment import DrugOptimizationEnv


class QLearningAgent:
    """
    Simple Q-Learning agent for drug optimization.

    This is a toy example. For production scale, MILES would enable:
    - Deep Q-Networks (DQN) instead of tabular Q-learning
    - Mixture of Experts for handling diverse chemical spaces
    - Distributed training for large compound libraries
    """

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """Initialize Q-learning agent."""
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: maps (discretized_state, action) -> Q-value
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        # Statistics
        self.training_rewards = []
        self.training_losses = []

    def _discretize_state(self, obs: np.ndarray) -> tuple:
        """
        Discretize state for tabular Q-learning.

        Supports both:
        - Vector observations from the original environment
        - Scalar / dummy observations (bandit-style) from the enhanced env
        """
        # Convert to 1D array for flexible handling
        obs_arr = np.array(obs).reshape(-1)

        # Bandit-style / dummy observation (e.g., env returns just 0)
        # Collapse everything into a single bucketed state.
        if obs_arr.size < 5:
            return (0, 0, 0, 0)

        # Original environment: use feature-based discretization
        compound_idx = int(obs_arr[0])
        promiscuity_bin = int(obs_arr[1] // 20)   # Bins of 20 hits
        cytotox_bin = int(obs_arr[2] * 10)        # 0.1 probability bins
        is_active = int(obs_arr[4])

        return (compound_idx, promiscuity_bin, cytotox_bin, is_active)

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            obs: Current observation
            training: If True, use exploration; if False, exploit only

        Returns:
            Selected action index
        """
        state = self._discretize_state(obs)

        # Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action
            q_values = self.q_table[state]
            return np.argmax(q_values)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> float:
        """
        Update Q-table using Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated

        Returns:
            TD error (for monitoring)
        """
        state = self._discretize_state(obs)
        next_state = self._discretize_state(next_obs)

        # Current Q-value
        current_q = self.q_table[state][action]

        # Target Q-value
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * next_max_q

        # TD error
        td_error = target_q - current_q

        # Update Q-value
        self.q_table[state][action] += self.lr * td_error

        return abs(td_error)

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save agent to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'training_rewards': self.training_rewards,
                'training_losses': self.training_losses
            }, f)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
            self.epsilon = data['epsilon']
            self.training_rewards = data['training_rewards']
            self.training_losses = data['training_losses']
        print(f"Agent loaded from {filepath}")


def train_agent(
    env: DrugOptimizationEnv,
    agent: QLearningAgent,
    n_episodes: int = 500,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """
    Train Q-learning agent.

    Args:
        env: Drug optimization environment
        agent: Q-learning agent
        n_episodes: Number of training episodes
        verbose: Print progress

    Returns:
        Training statistics
    """
    episode_rewards = []
    episode_lengths = []
    td_errors = []

    print(f"\n{'='*80}")
    print(f"TRAINING Q-LEARNING AGENT")
    print(f"{'='*80}")
    print(f"Episodes: {n_episodes}")
    print(f"Learning rate: {agent.lr}")
    print(f"Discount factor: {agent.gamma}")
    print(f"Epsilon: {agent.epsilon:.3f} → {agent.epsilon_end:.3f}")
    print(f"{'='*80}\n")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_td_errors = []
        step_count = 0

        while True:
            # Select action
            action = agent.select_action(obs, training=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update agent
            td_error = agent.update(obs, action, reward, next_obs, done)
            episode_td_errors.append(td_error)

            # Update state
            obs = next_obs
            episode_reward += reward
            step_count += 1

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        td_errors.append(np.mean(episode_td_errors))

        agent.training_rewards.append(episode_reward)
        agent.training_losses.append(np.mean(episode_td_errors))

        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_td = np.mean(td_errors[-50:])
            print(f"Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_reward:7.3f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Avg TD Error: {avg_td:6.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Final average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.3f}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Total Q-table entries: {len(agent.q_table)}")
    print(f"{'='*80}\n")

    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'td_errors': td_errors
    }


def evaluate_agent(
    env: DrugOptimizationEnv,
    agent: QLearningAgent,
    n_episodes: int = 10
) -> Dict[str, any]:
    """
    Evaluate trained agent.

    Args:
        env: Drug optimization environment
        agent: Trained agent
        n_episodes: Number of evaluation episodes

    Returns:
        Evaluation statistics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING TRAINED AGENT")
    print(f"{'='*80}")

    episode_rewards = []
    best_compounds = []
    compound_features = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        compounds_visited = []

        while True:
            # Select action (no exploration)
            action = agent.select_action(obs, training=False)

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            compounds_visited.append((info['compound_id'], info['features'], reward))

            if done:
                break

        episode_rewards.append(episode_reward)

        # Find best compound in this episode
        best_idx = np.argmax([r for _, _, r in compounds_visited])
        best_compound = compounds_visited[best_idx]
        best_compounds.append(best_compound)

        print(f"\nEpisode {episode+1}:")
        print(f"  Total Reward: {episode_reward:.3f}")
        print(f"  Best Compound: {best_compound[0]}")
        print(f"    Target Activity: {best_compound[1]['target_activity']:.2f}")
        print(f"    Promiscuity: {best_compound[1]['promiscuity']:.0f} hits")
        print(f"    Cytotoxicity: {best_compound[1]['cytotox_prob']:.2%}")
        print(f"    Reward: {best_compound[2]:.3f}")

    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Average Reward: {avg_reward:.3f} ± {std_reward:.3f}")
    print(f"Best Episode Reward: {max(episode_rewards):.3f}")
    print(f"Worst Episode Reward: {min(episode_rewards):.3f}")
    print(f"{'='*80}\n")

    return {
        'episode_rewards': episode_rewards,
        'best_compounds': best_compounds,
        'avg_reward': avg_reward,
        'std_reward': std_reward
    }


def plot_training_results(stats: Dict[str, List[float]], save_path: str = None):
    """Plot training statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Moving average window
    window = 50

    # 1. Episode Rewards
    ax = axes[0, 0]
    rewards = stats['episode_rewards']
    ax.plot(rewards, alpha=0.3, label='Raw')
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg, label=f'{window}-episode MA', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Training Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Episode Lengths
    ax = axes[0, 1]
    lengths = stats['episode_lengths']
    ax.plot(lengths, alpha=0.3, label='Raw')
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(lengths)), moving_avg, label=f'{window}-episode MA', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (steps)')
    ax.set_title('Episode Lengths Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. TD Errors
    ax = axes[1, 0]
    td_errors = stats['td_errors']
    ax.plot(td_errors, alpha=0.3, label='Raw')
    if len(td_errors) >= window:
        moving_avg = np.convolve(td_errors, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(td_errors)), moving_avg, label=f'{window}-episode MA', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Mean TD Error')
    ax.set_title('TD Errors Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Reward Distribution
    ax = axes[1, 1]
    ax.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
    ax.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")

    plt.show()


def main():
    """Main training pipeline."""
    print("\n" + "="*80)
    print("DRUG OPTIMIZATION RL - TRAINING PIPELINE")
    print("="*80)

    # Initialize environment
    env = DrugOptimizationEnv(
        drug_target_data_path='/Users/paulamerigojr.iipajo/Downloads/Drug Target Activity.csv',
        promiscuity_data_path='/Users/paulamerigojr.iipajo/Downloads/discovery2-results/discovery2_promiscuity_scores.csv',
        cytotox_model_path='/Users/paulamerigojr.iipajo/Downloads/discovery2-cytotoxicity-models/cubic_logistic_model.pkl',
        target_gene='BTK',
        max_steps=20,
        efficacy_weight=0.4,
        safety_weight=0.4,
        selectivity_weight=0.2
    )

    # Initialize agent
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    # Train agent
    training_stats = train_agent(env, agent, n_episodes=500, verbose=True)

    # Save agent
    agent.save('/Users/paulamerigojr.iipajo/drug_rl_agent.pkl')

    # Plot results
    plot_training_results(
        training_stats,
        save_path='/Users/paulamerigojr.iipajo/drug_rl_training_results.png'
    )

    # Evaluate agent
    eval_stats = evaluate_agent(env, agent, n_episodes=10)

    # Compare with random baseline
    print("\n" + "="*80)
    print("COMPARING WITH RANDOM BASELINE")
    print("="*80)

    random_rewards = []
    for _ in range(10):
        obs, info = env.reset()
        episode_reward = 0
        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        random_rewards.append(episode_reward)

    print(f"Random Policy Average Reward: {np.mean(random_rewards):.3f} ± {np.std(random_rewards):.3f}")
    print(f"Trained Agent Average Reward: {eval_stats['avg_reward']:.3f} ± {eval_stats['std_reward']:.3f}")
    print(f"Improvement: {((eval_stats['avg_reward'] - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
