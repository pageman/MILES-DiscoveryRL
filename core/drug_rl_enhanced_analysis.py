#!/usr/bin/env python3
"""
Enhanced Drug RL Training Analysis
===================================
Comprehensive analysis of Q-learning training results with:
- Learning curve analysis
- Convergence diagnostics
- Compound portfolio optimization
- Multi-target comparison
- Statistical hypothesis testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class DrugRLAnalyzer:
    """Comprehensive analyzer for Drug RL training results"""

    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path.home()
        self.analysis_results = {}

    def analyze_training_convergence(self, rewards: List[float],
                                     window: int = 20) -> Dict:
        """
        Analyze training convergence with multiple metrics

        Args:
            rewards: List of episode rewards
            window: Moving average window size

        Returns:
            Dictionary with convergence metrics
        """
        rewards = np.array(rewards)
        n_episodes = len(rewards)

        # Moving average
        ma = pd.Series(rewards).rolling(window=window, min_periods=1).mean().values

        # Moving standard deviation
        mstd = pd.Series(rewards).rolling(window=window, min_periods=1).std().values

        # Detect convergence (when variance stabilizes)
        variance_changes = np.diff(mstd)
        convergence_point = None

        if len(variance_changes) > window:
            # Find where variance change drops below threshold
            threshold = np.std(variance_changes) * 0.1
            stable_points = np.where(np.abs(variance_changes) < threshold)[0]
            if len(stable_points) > 0:
                convergence_point = stable_points[0]

        # Statistical tests
        first_quarter = rewards[:n_episodes//4]
        last_quarter = rewards[-n_episodes//4:]

        t_stat, p_value = stats.ttest_ind(last_quarter, first_quarter)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(first_quarter)**2 + np.std(last_quarter)**2) / 2)
        cohens_d = (np.mean(last_quarter) - np.mean(first_quarter)) / pooled_std if pooled_std > 0 else 0

        return {
            'n_episodes': n_episodes,
            'initial_reward': float(rewards[0]),
            'final_reward': float(rewards[-1]),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'convergence_episode': int(convergence_point) if convergence_point else None,
            'improvement_percent': float((rewards[-1] - rewards[0]) / abs(rewards[0]) * 100) if rewards[0] != 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'moving_average': ma.tolist(),
            'moving_std': mstd.tolist(),
            'significantly_improved': p_value < 0.05 and cohens_d > 0.5
        }

    def plot_comprehensive_training_curves(self, training_results: Dict[str, List[float]],
                                          save_path: Path = None):
        """
        Create comprehensive training visualization with 4 subplots
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Drug RL Training Analysis', fontsize=16, fontweight='bold')

        for name, rewards in training_results.items():
            rewards = np.array(rewards)
            episodes = np.arange(len(rewards))

            # 1. Raw rewards + moving average
            ax1 = axes[0, 0]
            ax1.plot(episodes, rewards, alpha=0.3, label=f'{name} (raw)', linewidth=1)
            ma = pd.Series(rewards).rolling(window=20).mean()
            ax1.plot(episodes, ma, label=f'{name} (MA-20)', linewidth=2)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Total Reward')
            ax1.set_title('Training Rewards Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Cumulative reward
            ax2 = axes[0, 1]
            cumulative = np.cumsum(rewards)
            ax2.plot(episodes, cumulative, label=f'{name}', linewidth=2)
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Cumulative Reward')
            ax2.set_title('Cumulative Reward Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. Reward distribution
            ax3 = axes[1, 0]
            ax3.hist(rewards, bins=30, alpha=0.5, label=f'{name}', density=True)
            ax3.axvline(np.mean(rewards), color='red', linestyle='--',
                       label=f'Mean: {np.mean(rewards):.2f}')
            ax3.set_xlabel('Reward')
            ax3.set_ylabel('Density')
            ax3.set_title('Reward Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. Variance over time
            ax4 = axes[1, 1]
            rolling_var = pd.Series(rewards).rolling(window=20).var()
            ax4.plot(episodes, rolling_var, label=f'{name}', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Variance (window=20)')
            ax4.set_title('Reward Variance Over Time (Stability)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comprehensive training plot to {save_path}")
        else:
            plt.show()

        return fig

    def analyze_compound_portfolio(self, env, agent, top_k: int = 10) -> pd.DataFrame:
        """
        Analyze the top-k compounds discovered by the agent

        Args:
            env: Drug optimization environment
            agent: Trained Q-learning agent
            top_k: Number of top compounds to analyze

        Returns:
            DataFrame with compound analysis
        """
        # Get Q-values for state 0 (single-state environment)
        q_values = agent.q_table[0]

        # Get top-k compounds by Q-value
        top_indices = np.argsort(q_values)[-top_k:][::-1]

        compound_data = []

        for rank, idx in enumerate(top_indices, 1):
            compound_id = env.compounds[idx]
            q_value = q_values[idx]

            # Get compound details
            efficacy_data = env.drug_target_df[
                env.drug_target_df['compound_id'] == compound_id
            ]

            if not efficacy_data.empty:
                efficacy = efficacy_data['outcome_max_activity'].iloc[0]
                is_active = efficacy_data.get('outcome_is_active', [False]).iloc[0]
            else:
                efficacy = 0.0
                is_active = False

            promiscuity = env.promiscuity_scores.get(compound_id, 0.0)

            # Estimate safety (inverse of toxicity prediction)
            try:
                safety_pred = env.cytotox_model.predict_proba(
                    np.array([[promiscuity]])
                )[:, 1][0]
                safety = 1.0 - safety_pred
            except:
                safety = 0.5

            compound_data.append({
                'rank': rank,
                'compound_id': compound_id,
                'q_value': float(q_value),
                'efficacy': float(efficacy),
                'safety': float(safety),
                'promiscuity': float(promiscuity),
                'selectivity': float(1.0 - promiscuity / env.promiscuity_df['promiscuity_score'].max()),
                'is_active': bool(is_active),
                'composite_score': float(
                    0.4 * efficacy / 100.0 +  # Normalize to 0-1
                    0.4 * safety +
                    0.2 * (1.0 - promiscuity / env.promiscuity_df['promiscuity_score'].max())
                )
            })

        df = pd.DataFrame(compound_data)
        return df

    def plot_compound_portfolio(self, portfolio_df: pd.DataFrame, save_path: Path = None):
        """Create visualization of compound portfolio"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Compound Portfolio Analysis', fontsize=16, fontweight='bold')

        # 1. Efficacy-Safety scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(portfolio_df['efficacy'], portfolio_df['safety'],
                             s=portfolio_df['q_value']*100,
                             c=portfolio_df['promiscuity'],
                             cmap='RdYlGn_r', alpha=0.6)

        for idx, row in portfolio_df.iterrows():
            ax1.annotate(f"{row['rank']}",
                        (row['efficacy'], row['safety']),
                        fontsize=8, ha='center')

        ax1.set_xlabel('Efficacy (max activity)')
        ax1.set_ylabel('Safety (1 - toxicity)')
        ax1.set_title('Efficacy-Safety Trade-off')
        plt.colorbar(scatter, ax=ax1, label='Promiscuity')
        ax1.grid(True, alpha=0.3)

        # 2. Radar chart for top 5 compounds
        ax2 = axes[0, 1]
        categories = ['Efficacy', 'Safety', 'Selectivity', 'Q-Value']

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        ax2 = plt.subplot(2, 2, 2, projection='polar')

        for idx, row in portfolio_df.head(5).iterrows():
            values = [
                row['efficacy'] / 100.0,  # Normalize
                row['safety'],
                row['selectivity'],
                row['q_value'] / portfolio_df['q_value'].max()  # Normalize
            ]
            values += values[:1]

            ax2.plot(angles, values, 'o-', linewidth=2,
                    label=f"Rank {row['rank']}: {row['compound_id'][:8]}")
            ax2.fill(angles, values, alpha=0.15)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 1)
        ax2.set_title('Top 5 Compounds - Multi-dimensional Profile')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)

        # 3. Bar chart of composite scores
        ax3 = axes[1, 0]
        colors = ['green' if x else 'orange' for x in portfolio_df['is_active']]
        ax3.barh(portfolio_df['rank'], portfolio_df['composite_score'], color=colors, alpha=0.7)
        ax3.set_xlabel('Composite Score')
        ax3.set_ylabel('Rank')
        ax3.set_title('Composite Score by Rank (Green=Active, Orange=Inactive)')
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='x')

        # 4. Correlation heatmap
        ax4 = axes[1, 1]
        corr_cols = ['q_value', 'efficacy', 'safety', 'promiscuity', 'composite_score']
        corr = portfolio_df[corr_cols].corr()

        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax4, cbar_kws={'label': 'Correlation'})
        ax4.set_title('Feature Correlation Matrix')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved portfolio plot to {save_path}")
        else:
            plt.show()

        return fig

    def compare_agents(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple agents/configurations

        Args:
            results: Dict mapping agent name to results dict with 'rewards' key

        Returns:
            Comparison DataFrame
        """
        comparison = []

        for name, result in results.items():
            rewards = result['rewards']
            convergence = self.analyze_training_convergence(rewards)

            comparison.append({
                'agent': name,
                'final_reward': convergence['final_reward'],
                'mean_reward': convergence['mean_reward'],
                'std_reward': convergence['std_reward'],
                'max_reward': convergence['max_reward'],
                'improvement_%': convergence['improvement_percent'],
                'cohens_d': convergence['cohens_d'],
                'converged': convergence['convergence_episode'] is not None,
                'convergence_episode': convergence.get('convergence_episode', 'N/A')
            })

        return pd.DataFrame(comparison).sort_values('mean_reward', ascending=False)

    def statistical_comparison(self, results: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Perform pairwise statistical comparisons between agents

        Returns:
            DataFrame with pairwise t-test results
        """
        agent_names = list(results.keys())
        n_agents = len(agent_names)

        comparisons = []

        for i in range(n_agents):
            for j in range(i+1, n_agents):
                name1, name2 = agent_names[i], agent_names[j]
                rewards1, rewards2 = results[name1], results[name2]

                # T-test
                t_stat, p_value = stats.ttest_ind(rewards1, rewards2)

                # Effect size
                pooled_std = np.sqrt((np.std(rewards1)**2 + np.std(rewards2)**2) / 2)
                cohens_d = (np.mean(rewards1) - np.mean(rewards2)) / pooled_std

                # Mann-Whitney U (non-parametric)
                u_stat, u_pvalue = stats.mannwhitneyu(rewards1, rewards2, alternative='two-sided')

                comparisons.append({
                    'agent_1': name1,
                    'agent_2': name2,
                    'mean_diff': np.mean(rewards1) - np.mean(rewards2),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'mann_whitney_u': u_stat,
                    'mann_whitney_p': u_pvalue,
                    'significant': p_value < 0.05,
                    'effect_size': 'large' if abs(cohens_d) > 0.8 else ('medium' if abs(cohens_d) > 0.5 else 'small')
                })

        return pd.DataFrame(comparisons)


def run_comprehensive_analysis(training_results: Dict = None,
                               env = None,
                               agent = None,
                               output_dir: Path = None):
    """
    Run complete analysis suite

    Args:
        training_results: Dict mapping config name to training stats
        env: Drug optimization environment
        agent: Trained agent
        output_dir: Directory to save outputs
    """
    output_dir = output_dir or Path.home() / 'drug_rl_analysis'
    output_dir.mkdir(exist_ok=True, parents=True)

    analyzer = DrugRLAnalyzer(output_dir)

    print("="*80)
    print("COMPREHENSIVE DRUG RL TRAINING ANALYSIS")
    print("="*80)

    results_summary = {}

    if training_results:
        print("\n1. ANALYZING TRAINING CONVERGENCE")
        print("-" * 80)

        for name, result in training_results.items():
            print(f"\nAnalyzing: {name}")
            convergence = analyzer.analyze_training_convergence(result['rewards'])
            results_summary[name] = convergence

            print(f"  Initial Reward: {convergence['initial_reward']:.3f}")
            print(f"  Final Reward: {convergence['final_reward']:.3f}")
            print(f"  Mean ± Std: {convergence['mean_reward']:.3f} ± {convergence['std_reward']:.3f}")
            print(f"  Improvement: {convergence['improvement_percent']:.1f}%")
            print(f"  Cohen's d: {convergence['cohens_d']:.3f}")
            print(f"  Significantly Improved: {convergence['significantly_improved']}")

            if convergence['convergence_episode']:
                print(f"  Converged at episode: {convergence['convergence_episode']}")

        # Save convergence metrics
        pd.DataFrame(results_summary).T.to_csv(
            output_dir / 'convergence_metrics.csv'
        )
        print(f"\n✓ Saved convergence metrics to {output_dir / 'convergence_metrics.csv'}")

        # Plot comprehensive training curves
        print("\n2. GENERATING TRAINING VISUALIZATIONS")
        print("-" * 80)
        rewards_dict = {name: result['rewards'] for name, result in training_results.items()}
        analyzer.plot_comprehensive_training_curves(
            rewards_dict,
            save_path=output_dir / 'comprehensive_training_analysis.png'
        )

        # Statistical comparison
        if len(training_results) > 1:
            print("\n3. STATISTICAL COMPARISON")
            print("-" * 80)
            comparison_df = analyzer.compare_agents(training_results)
            print(comparison_df.to_string(index=False))
            comparison_df.to_csv(output_dir / 'agent_comparison.csv', index=False)
            print(f"\n✓ Saved to {output_dir / 'agent_comparison.csv'}")

            stat_comp = analyzer.statistical_comparison(rewards_dict)
            print("\nPairwise Statistical Tests:")
            print(stat_comp.to_string(index=False))
            stat_comp.to_csv(output_dir / 'statistical_comparison.csv', index=False)
            print(f"✓ Saved to {output_dir / 'statistical_comparison.csv'}")

    if env and agent:
        print("\n4. COMPOUND PORTFOLIO ANALYSIS")
        print("-" * 80)
        portfolio = analyzer.analyze_compound_portfolio(env, agent, top_k=20)
        print("\nTop 20 Compounds:")
        print(portfolio.to_string(index=False))

        portfolio.to_csv(output_dir / 'top_compounds_portfolio.csv', index=False)
        print(f"\n✓ Saved to {output_dir / 'top_compounds_portfolio.csv'}")

        analyzer.plot_compound_portfolio(
            portfolio,
            save_path=output_dir / 'compound_portfolio_analysis.png'
        )

    # Save complete analysis report
    with open(output_dir / 'analysis_summary.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_summary = {
            k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                for kk, vv in v.items() if kk not in ['moving_average', 'moving_std']}
            for k, v in results_summary.items()
        }
        json.dump(json_summary, f, indent=2)

    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE! All outputs saved to: {output_dir}")
    print("="*80)

    return analyzer, results_summary


if __name__ == "__main__":
    print("Drug RL Enhanced Analysis Module")
    print("Import this module and use run_comprehensive_analysis()")
    print("\nExample:")
    print("  from drug_rl_enhanced_analysis import run_comprehensive_analysis")
    print("  analyzer, summary = run_comprehensive_analysis(")
    print("      training_results={'Q-Learning': {'rewards': [...]},")
    print("      env=env, agent=agent")
    print("  )")
