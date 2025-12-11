#!/usr/bin/env python3
"""
Drug Optimization RL Environment
=================================

A toy reinforcement learning environment for drug discovery that balances:
- Target activity (efficacy)
- Cytotoxicity risk (safety)

This demonstrates RL concepts applicable to drug optimization and could be
scaled using MILES framework for production workloads.

Environment:
- State: Compound features (promiscuity scores, target activities)
- Action: Select next compound or modify properties
- Reward: Balance efficacy and safety

Author: Generated for RL-based drug discovery demonstration
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib
import json
from typing import Dict, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')


class DrugOptimizationEnv(gym.Env):
    """
    Gymnasium environment for drug optimization.

    The agent's goal is to find compounds with:
    1. High activity against target(s)
    2. Low cytotoxicity risk
    3. Low promiscuity (good selectivity)

    This is a multi-objective optimization problem common in drug discovery.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        drug_target_data_path: str,
        promiscuity_data_path: str,
        cytotox_model_path: str,
        target_gene: str = 'BTK',
        max_steps: int = 50,
        efficacy_weight: float = 0.4,
        safety_weight: float = 0.4,
        selectivity_weight: float = 0.2
    ):
        """
        Initialize the drug optimization environment.

        Args:
            drug_target_data_path: Path to Drug Target Activity.csv
            promiscuity_data_path: Path to promiscuity scores CSV
            cytotox_model_path: Path to cytotoxicity prediction model
            target_gene: Target gene of interest (e.g., 'BTK')
            max_steps: Maximum steps per episode
            efficacy_weight: Weight for efficacy in reward (0-1)
            safety_weight: Weight for safety in reward (0-1)
            selectivity_weight: Weight for selectivity in reward (0-1)
        """
        super().__init__()

        # Load data
        print(f"Loading drug-target activity data...")
        self.drug_data = pd.read_csv(drug_target_data_path)

        print(f"Loading promiscuity scores...")
        self.promiscuity_data = pd.read_csv(promiscuity_data_path)

        print(f"Loading cytotoxicity model...")
        self.cytotox_model = joblib.load(cytotox_model_path)

        # Filter for target of interest
        self.target_gene = target_gene
        self.target_data = self.drug_data[
            self.drug_data['target__gene'] == target_gene
        ].copy()

        # Get unique compounds
        self.compound_ids = self.target_data['compound_id'].unique()
        self.n_compounds = len(self.compound_ids)

        print(f"Environment initialized:")
        print(f"  Target: {target_gene}")
        print(f"  Compounds: {self.n_compounds}")
        print(f"  Max steps: {max_steps}")

        # Create compound index mapping
        self.compound_to_idx = {cid: idx for idx, cid in enumerate(self.compound_ids)}

        # Environment parameters
        self.max_steps = max_steps
        self.efficacy_weight = efficacy_weight
        self.safety_weight = safety_weight
        self.selectivity_weight = selectivity_weight

        # State space: [current_compound_idx, promiscuity, cytotox_prob,
        #                target_activity, is_active, steps_remaining]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -200, 0, 0]),
            high=np.array([self.n_compounds, 300, 1.0, 200, 1, max_steps]),
            dtype=np.float32
        )

        # Action space: Select compound index
        self.action_space = spaces.Discrete(self.n_compounds)

        # Episode state
        self.current_step = 0
        self.current_compound_idx = None
        self.visited_compounds = set()
        self.episode_rewards = []

        # Statistics for normalization
        self.activity_mean = self.target_data['outcome_max_activity'].mean()
        self.activity_std = self.target_data['outcome_max_activity'].std()

    def _get_compound_features(self, compound_id: str) -> Dict[str, float]:
        """Get features for a compound."""
        # Get target activity
        compound_data = self.target_data[
            self.target_data['compound_id'] == compound_id
        ].iloc[0]

        target_activity = compound_data['outcome_max_activity']
        is_active = float(compound_data['outcome_is_active'])

        # Get promiscuity score
        prom_data = self.promiscuity_data[
            self.promiscuity_data['compound_id'] == compound_id
        ]

        if len(prom_data) > 0:
            promiscuity = prom_data.iloc[0]['overall_promiscuity']
            kinase_prom = prom_data.iloc[0]['kinase_promiscuity']
        else:
            promiscuity = 0
            kinase_prom = 0

        # Predict cytotoxicity
        cytotox_prob = self._predict_cytotoxicity(promiscuity)

        return {
            'target_activity': target_activity,
            'is_active': is_active,
            'promiscuity': promiscuity,
            'kinase_promiscuity': kinase_prom,
            'cytotox_prob': cytotox_prob
        }

    def _predict_cytotoxicity(self, promiscuity: float) -> float:
        """Predict cytotoxicity probability from promiscuity."""
        try:
            from statsmodels.tools import add_constant
            # Create cubic features
            X = np.array([[promiscuity, promiscuity**2, promiscuity**3]])
            X_with_const = add_constant(X)
            prob = self.cytotox_model.predict(X_with_const)[0]
            return np.clip(prob, 0, 1)
        except Exception as e:
            # Fallback to simple sigmoid if model fails
            return 1 / (1 + np.exp(-0.05 * (promiscuity - 50)))

    def _compute_reward(
        self,
        target_activity: float,
        is_active: bool,
        promiscuity: float,
        cytotox_prob: float,
        is_new_compound: bool
    ) -> float:
        """
        Compute reward balancing efficacy, safety, and selectivity.

        Reward components:
        1. Efficacy: High target activity
        2. Safety: Low cytotoxicity probability
        3. Selectivity: Low promiscuity

        Returns:
            Scalar reward in range [-1, 1]
        """
        # Efficacy component (0 to 1)
        # Normalize activity and apply sigmoid
        if is_active:
            normalized_activity = (target_activity - self.activity_mean) / (self.activity_std + 1e-6)
            efficacy_score = 1 / (1 + np.exp(-normalized_activity))
        else:
            efficacy_score = 0.0

        # Safety component (0 to 1)
        # Lower cytotoxicity is better
        safety_score = 1 - cytotox_prob

        # Selectivity component (0 to 1)
        # Lower promiscuity is better
        # 50 hits is median, >100 is very promiscuous
        selectivity_score = 1 / (1 + np.exp(0.05 * (promiscuity - 50)))

        # Combine components
        reward = (
            self.efficacy_weight * efficacy_score +
            self.safety_weight * safety_score +
            self.selectivity_weight * selectivity_score
        )

        # Bonus for exploring new compounds
        if is_new_compound:
            reward += 0.05

        # Penalty for revisiting compounds
        if not is_new_compound:
            reward -= 0.1

        return reward

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Reset episode state
        self.current_step = 0
        self.visited_compounds = set()
        self.episode_rewards = []

        # Start with random compound
        self.current_compound_idx = self.np_random.integers(0, self.n_compounds)
        compound_id = self.compound_ids[self.current_compound_idx]
        self.visited_compounds.add(compound_id)

        # Get features
        features = self._get_compound_features(compound_id)

        # Construct observation
        obs = np.array([
            self.current_compound_idx,
            features['promiscuity'],
            features['cytotox_prob'],
            features['target_activity'],
            features['is_active'],
            self.max_steps - self.current_step
        ], dtype=np.float32)

        info = {
            'compound_id': compound_id,
            'features': features
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Compound index to select

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Get compound from action
        compound_id = self.compound_ids[action]
        is_new_compound = compound_id not in self.visited_compounds
        self.visited_compounds.add(compound_id)

        # Get features
        features = self._get_compound_features(compound_id)

        # Compute reward
        reward = self._compute_reward(
            target_activity=features['target_activity'],
            is_active=features['is_active'],
            promiscuity=features['promiscuity'],
            cytotox_prob=features['cytotox_prob'],
            is_new_compound=is_new_compound
        )

        self.episode_rewards.append(reward)

        # Update state
        self.current_compound_idx = action

        # Construct observation
        obs = np.array([
            self.current_compound_idx,
            features['promiscuity'],
            features['cytotox_prob'],
            features['target_activity'],
            features['is_active'],
            self.max_steps - self.current_step
        ], dtype=np.float32)

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        # Info
        info = {
            'compound_id': compound_id,
            'features': features,
            'is_new': is_new_compound,
            'cumulative_reward': sum(self.episode_rewards)
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment state (human-readable)."""
        if self.current_compound_idx is not None:
            compound_id = self.compound_ids[self.current_compound_idx]
            features = self._get_compound_features(compound_id)

            print(f"\n{'='*60}")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Current Compound: {compound_id}")
            print(f"{'='*60}")
            print(f"  Target ({self.target_gene}) Activity: {features['target_activity']:.2f}")
            print(f"  Is Active: {features['is_active']}")
            print(f"  Promiscuity: {features['promiscuity']:.0f} hits")
            print(f"  Kinase Promiscuity: {features['kinase_promiscuity']:.0f} hits")
            print(f"  Cytotoxicity Prob: {features['cytotox_prob']:.2%}")
            print(f"  Compounds Visited: {len(self.visited_compounds)}/{self.n_compounds}")
            if self.episode_rewards:
                print(f"  Episode Reward: {sum(self.episode_rewards):.3f}")
            print(f"{'='*60}\n")


def demonstrate_environment():
    """Demonstrate the environment with random policy."""
    print("="*80)
    print("DRUG OPTIMIZATION RL ENVIRONMENT - DEMONSTRATION")
    print("="*80)

    # Initialize environment
    env = DrugOptimizationEnv(
        drug_target_data_path='/Users/paulamerigojr.iipajo/Downloads/Drug Target Activity.csv',
        promiscuity_data_path='/Users/paulamerigojr.iipajo/Downloads/discovery2-results/discovery2_promiscuity_scores.csv',
        cytotox_model_path='/Users/paulamerigojr.iipajo/Downloads/discovery2-cytotoxicity-models/cubic_logistic_model.pkl',
        target_gene='BTK',
        max_steps=10
    )

    print("\n" + "="*80)
    print("RUNNING RANDOM POLICY (Baseline)")
    print("="*80)

    # Run one episode with random policy
    obs, info = env.reset(seed=42)
    env.render()

    for step in range(10):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action: Selected compound index {action}")
        print(f"Reward: {reward:.3f}")
        env.render()

        if terminated or truncated:
            break

    print(f"\nEpisode finished!")
    print(f"Total reward: {info['cumulative_reward']:.3f}")
    print(f"Compounds explored: {len(env.visited_compounds)}/{env.n_compounds}")

    return env


if __name__ == '__main__':
    env = demonstrate_environment()
