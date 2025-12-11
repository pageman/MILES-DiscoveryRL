#!/usr/bin/env python3
# Copyright 2024 Paul Pajo and Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MILES-Inspired Drug RL Concepts
================================

This script demonstrates how MILES framework concepts could be applied
to drug optimization, showing the evolution from toy Q-learning to
production-scale MoE-based RL.

MILES Core Concepts Applied to Drug Discovery:
1. Mixture of Experts: Different experts for target classes
2. On-Policy Training: Alignment between training and serving
3. Distributed Rollouts: Parallel compound evaluation
4. Memory Robustness: Handle large compound libraries
5. Speculative Training: Fast compound generation

Author: Demonstration of MILES concepts for drug discovery
References: MILES framework (https://github.com/radixark/miles)
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class TargetClass(Enum):
    """Target classes for expert routing."""
    KINASE = "Kinase"
    GPCR = "7TM"  # G-Protein Coupled Receptors
    NUCLEAR_RECEPTOR = "NR"
    OTHER = "Other"


@dataclass
class CompoundFeatures:
    """Molecular representation for MILES system."""
    compound_id: str
    smiles: str
    morgan_fingerprint: np.ndarray  # 2048-bit
    target_activity: float
    promiscuity: int
    kinase_promiscuity: int
    gpcr_promiscuity: int
    nr_promiscuity: int
    cytotox_prob: float
    target_class: TargetClass


class DrugPolicyExpert:
    """
    Single expert in the MoE architecture.

    In MILES, each expert would be a full neural network.
    Here we show the concept with a simple policy.
    """

    def __init__(
        self,
        name: str,
        target_class: TargetClass,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        output_dim: int = 100
    ):
        """
        Initialize expert.

        Args:
            name: Expert identifier
            target_class: Which target class this expert handles
            input_dim: Input feature dimension (e.g., molecular fingerprint size)
            hidden_dim: Hidden layer size
            output_dim: Action space size
        """
        self.name = name
        self.target_class = target_class
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # In MILES: self.network = nn.Sequential(...)
        # Here: conceptual representation
        self.weights = self._init_weights()

        print(f"  Expert '{name}' initialized for {target_class.value}")
        print(f"    Architecture: {input_dim} → {hidden_dim} → {output_dim}")

    def _init_weights(self) -> Dict[str, np.ndarray]:
        """Initialize network weights (conceptual)."""
        return {
            'W1': np.random.randn(self.input_dim, self.hidden_dim) * 0.01,
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.randn(self.hidden_dim, self.output_dim) * 0.01,
            'b2': np.zeros(self.output_dim)
        }

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through expert network.

        Args:
            features: Input features (molecular fingerprint)

        Returns:
            Action logits
        """
        # Layer 1
        h1 = np.dot(features, self.weights['W1']) + self.weights['b1']
        h1 = np.maximum(0, h1)  # ReLU

        # Layer 2
        logits = np.dot(h1, self.weights['W2']) + self.weights['b2']

        return logits

    def get_load_balancing_loss(self) -> float:
        """
        Compute expert load balancing loss (MILES concept).

        Ensures experts are used evenly to prevent collapse.
        """
        # In MILES: track routing frequency
        return 0.0  # Conceptual


class MoERouter:
    """
    Router network for Mixture of Experts.

    In MILES, this learns which expert to use for each input.
    """

    def __init__(self, n_experts: int, input_dim: int = 2048):
        """Initialize router."""
        self.n_experts = n_experts
        self.input_dim = input_dim

        # Router weights (conceptual)
        self.W_router = np.random.randn(input_dim, n_experts) * 0.01
        self.b_router = np.zeros(n_experts)

        print(f"  Router initialized: {input_dim} → {n_experts} experts")

    def route(self, features: np.ndarray) -> np.ndarray:
        """
        Compute routing weights for each expert.

        Args:
            features: Input features

        Returns:
            Expert weights (softmax over experts)
        """
        logits = np.dot(features, self.W_router) + self.b_router

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / np.sum(exp_logits)

        return weights

    def top_k_routing(self, features: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Top-k routing: use only top k experts (sparse MoE).

        Args:
            features: Input features
            k: Number of experts to activate

        Returns:
            (top_k_indices, top_k_weights)
        """
        weights = self.route(features)
        top_k_idx = np.argsort(weights)[-k:]
        top_k_weights = weights[top_k_idx]

        # Renormalize
        top_k_weights = top_k_weights / np.sum(top_k_weights)

        return top_k_idx, top_k_weights


class DrugMixtureOfExperts:
    """
    MILES-style Mixture of Experts for drug optimization.

    Architecture:
    - Multiple expert networks (one per target class)
    - Router network selects expert(s) based on input
    - Load balancing to prevent expert collapse
    """

    def __init__(
        self,
        n_actions: int = 100,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        use_sparse_routing: bool = True,
        top_k: int = 2
    ):
        """
        Initialize MoE model.

        Args:
            n_actions: Action space size
            input_dim: Input feature dimension
            hidden_dim: Expert hidden dimension
            use_sparse_routing: Use top-k sparse routing
            top_k: Number of experts to activate
        """
        print("\n" + "="*80)
        print("INITIALIZING MIXTURE OF EXPERTS (MILES-STYLE)")
        print("="*80)

        self.n_actions = n_actions
        self.input_dim = input_dim
        self.use_sparse_routing = use_sparse_routing
        self.top_k = top_k

        # Initialize experts
        print("\nInitializing Experts:")
        self.experts = [
            DrugPolicyExpert("KinaseExpert", TargetClass.KINASE, input_dim, hidden_dim, n_actions),
            DrugPolicyExpert("GPCRExpert", TargetClass.GPCR, input_dim, hidden_dim, n_actions),
            DrugPolicyExpert("NRExpert", TargetClass.NUCLEAR_RECEPTOR, input_dim, hidden_dim, n_actions),
        ]

        # Initialize router
        print("\nInitializing Router:")
        self.router = MoERouter(len(self.experts), input_dim)

        print("\n" + "="*80)
        print(f"MoE Initialized: {len(self.experts)} experts, {n_actions} actions")
        if use_sparse_routing:
            print(f"Using sparse routing (top-{top_k})")
        print("="*80 + "\n")

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through MoE.

        MILES Implementation:
        1. Router computes expert weights
        2. Top-k experts are activated (sparse)
        3. Expert outputs are combined
        4. Load balancing loss is added

        Args:
            features: Input features (molecular fingerprint)

        Returns:
            Combined action logits
        """
        if self.use_sparse_routing:
            # Sparse MoE: activate only top-k experts
            expert_indices, expert_weights = self.router.top_k_routing(features, self.top_k)

            # Compute expert outputs
            outputs = []
            for idx, weight in zip(expert_indices, expert_weights):
                expert_output = self.experts[idx].forward(features)
                outputs.append(weight * expert_output)

            # Combine
            combined_output = np.sum(outputs, axis=0)

        else:
            # Dense MoE: use all experts
            expert_weights = self.router.route(features)

            # Compute expert outputs
            outputs = []
            for expert, weight in zip(self.experts, expert_weights):
                expert_output = expert.forward(features)
                outputs.append(weight * expert_output)

            # Combine
            combined_output = np.sum(outputs, axis=0)

        return combined_output

    def get_expert_statistics(self) -> Dict[str, any]:
        """Get statistics about expert usage."""
        return {
            'n_experts': len(self.experts),
            'expert_names': [e.name for e in self.experts],
            'routing_strategy': 'sparse' if self.use_sparse_routing else 'dense',
            'top_k': self.top_k if self.use_sparse_routing else len(self.experts)
        }


class MILESRolloutSystem:
    """
    MILES-style distributed rollout system for drug discovery.

    Concept: Separate training and rollout (generation/evaluation)
    - Training: Update MoE model on GPUs
    - Rollout: Generate compound evaluations in parallel
    - Data Buffer: Store experiences for training
    """

    def __init__(
        self,
        moe_model: DrugMixtureOfExperts,
        n_workers: int = 16,
        batch_size: int = 256
    ):
        """
        Initialize rollout system.

        Args:
            moe_model: MoE policy model
            n_workers: Number of parallel workers (SGLang concept)
            batch_size: Batch size for parallel evaluation
        """
        self.moe_model = moe_model
        self.n_workers = n_workers
        self.batch_size = batch_size

        print(f"\nMILES Rollout System Initialized:")
        print(f"  Workers: {n_workers}")
        print(f"  Batch size: {batch_size}")
        print(f"  Throughput: ~{n_workers * batch_size} compounds/batch")

    def parallel_rollout(
        self,
        compound_features: List[CompoundFeatures],
        n_steps: int = 10
    ) -> List[Tuple[CompoundFeatures, float]]:
        """
        Perform parallel rollout across workers.

        In MILES:
        - SGLang backend handles parallel inference
        - Each worker evaluates different compounds
        - Results are aggregated in data buffer

        Args:
            compound_features: Batch of compounds to evaluate
            n_steps: Steps per rollout

        Returns:
            List of (compound, reward) pairs
        """
        print(f"\n  Executing parallel rollout:")
        print(f"    Compounds: {len(compound_features)}")
        print(f"    Steps per rollout: {n_steps}")
        print(f"    Total evaluations: {len(compound_features) * n_steps}")

        results = []

        # Simulate parallel evaluation
        # In MILES: distributed across workers with SGLang
        for compound in compound_features:
            # Get policy action
            action_logits = self.moe_model.forward(compound.morgan_fingerprint)

            # Compute reward (conceptual)
            reward = self._compute_reward(compound)

            results.append((compound, reward))

        print(f"    ✓ Rollout complete")

        return results

    def _compute_reward(self, compound: CompoundFeatures) -> float:
        """Compute reward for compound (simplified)."""
        efficacy = compound.target_activity / 100.0
        safety = 1.0 - compound.cytotox_prob
        selectivity = 1.0 / (1.0 + np.exp(0.05 * (compound.promiscuity - 50)))

        reward = 0.4 * efficacy + 0.4 * safety + 0.2 * selectivity
        return reward


def demonstrate_miles_concepts():
    """Demonstrate MILES concepts applied to drug discovery."""
    print("\n" + "="*80)
    print("MILES FRAMEWORK CONCEPTS FOR DRUG DISCOVERY")
    print("="*80)

    print("\n📚 MILES Overview:")
    print("  - Enterprise RL framework for MoE models")
    print("  - Decoupled training and rollout")
    print("  - Production-ready distributed training")
    print("  - Designed for large-scale post-training")

    print("\n🧬 Application to Drug Discovery:")
    print("  - MoE experts specialize in target classes")
    print("  - Parallel compound evaluation")
    print("  - Multi-objective optimization")
    print("  - Scale to millions of molecules")

    # 1. Initialize MoE model
    print("\n" + "─"*80)
    print("STEP 1: INITIALIZE MIXTURE OF EXPERTS")
    print("─"*80)

    moe = DrugMixtureOfExperts(
        n_actions=100,
        input_dim=2048,
        hidden_dim=512,
        use_sparse_routing=True,
        top_k=2
    )

    # 2. Create example compound features
    print("\n" + "─"*80)
    print("STEP 2: CREATE EXAMPLE COMPOUNDS")
    print("─"*80)

    example_compounds = [
        CompoundFeatures(
            compound_id="EB001168",
            smiles="CC1=NC(NC2=NC=C(S2)C(=O)NC2=C(C)C=CC=C2Cl)=CC(=N1)N1CCN(CCO)CC1",
            morgan_fingerprint=np.random.rand(2048),  # Conceptual
            target_activity=99.8,
            promiscuity=115,
            kinase_promiscuity=33,
            gpcr_promiscuity=79,
            nr_promiscuity=2,
            cytotox_prob=0.85,
            target_class=TargetClass.KINASE
        ),
        CompoundFeatures(
            compound_id="EB000676",
            smiles="NC(=O)C1=C2NCC[C@@H](C3CCN(CC3)C(=O)C=C)N2N=C1C1=CC=C(OC2=CC=CC=C2)C=C1",
            morgan_fingerprint=np.random.rand(2048),
            target_activity=100.5,
            promiscuity=46,
            kinase_promiscuity=20,
            gpcr_promiscuity=19,
            nr_promiscuity=7,
            cytotox_prob=0.25,
            target_class=TargetClass.KINASE
        ),
    ]

    for comp in example_compounds:
        print(f"\n  Compound: {comp.compound_id}")
        print(f"    Target: {comp.target_class.value}")
        print(f"    Activity: {comp.target_activity:.1f}")
        print(f"    Promiscuity: {comp.promiscuity} hits")
        print(f"    Cytotoxicity: {comp.cytotox_prob:.2%}")

    # 3. MoE forward pass
    print("\n" + "─"*80)
    print("STEP 3: MOE FORWARD PASS")
    print("─"*80)

    for comp in example_compounds:
        print(f"\n  Processing {comp.compound_id}:")

        # Router decision
        expert_weights = moe.router.route(comp.morgan_fingerprint)
        print(f"    Router weights:")
        for expert, weight in zip(moe.experts, expert_weights):
            print(f"      {expert.name}: {weight:.3f}")

        # Top-k routing
        top_k_idx, top_k_weights = moe.router.top_k_routing(comp.morgan_fingerprint, k=2)
        print(f"    Top-2 experts:")
        for idx, weight in zip(top_k_idx, top_k_weights):
            print(f"      {moe.experts[idx].name}: {weight:.3f}")

        # Forward pass
        action_logits = moe.forward(comp.morgan_fingerprint)
        print(f"    Output logits shape: {action_logits.shape}")
        print(f"    Best action: {np.argmax(action_logits)}")

    # 4. Rollout system
    print("\n" + "─"*80)
    print("STEP 4: DISTRIBUTED ROLLOUT SYSTEM")
    print("─"*80)

    rollout_system = MILESRolloutSystem(
        moe_model=moe,
        n_workers=16,
        batch_size=256
    )

    # Execute rollout
    results = rollout_system.parallel_rollout(example_compounds, n_steps=10)

    print("\n  Rollout Results:")
    for compound, reward in results:
        print(f"    {compound.compound_id}: reward = {reward:.3f}")

    # 5. Key takeaways
    print("\n" + "="*80)
    print("KEY TAKEAWAYS: MILES FOR DRUG DISCOVERY")
    print("="*80)

    print("""
    ✓ Mixture of Experts:
      - Different experts for kinase vs GPCR vs NR targets
      - Router learns which expert to use
      - Prevents catastrophic forgetting across target classes

    ✓ Distributed Rollouts:
      - Parallel compound evaluation (16+ workers)
      - Batch processing (256+ compounds)
      - Throughput: 10K-100K compounds/second

    ✓ On-Policy Training:
      - Training policy matches rollout policy
      - Zero train-test mismatch
      - Better sample efficiency

    ✓ Memory Robustness:
      - FSDP handles large models
      - Graceful OOM error handling
      - Scale to millions of compounds

    ✓ Production Ready:
      - Stable, tested framework
      - Enterprise deployment
      - Real-world drug discovery applications
    """)

    print("="*80 + "\n")

    return moe, rollout_system


if __name__ == '__main__':
    demonstrate_miles_concepts()
