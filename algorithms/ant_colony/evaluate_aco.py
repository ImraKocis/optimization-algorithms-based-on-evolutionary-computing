from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithms.ant_colony.ant_colony import AntColonyOptimizer


@dataclass(frozen=True)
class ACOExperiment:
    distances: np.ndarray

    def run_parameter_study(self,
                            alphas,
                            betas,
                            evaporation_rates,
                            num_ants_multipliers,
                            n_trials=30,
                            num_iterations=500,
                            n_jobs=-1
                            ) -> pd.DataFrame:
        n_cities = len(self.distances)

        param_combinations = list(itertools.product(
            alphas,
            betas,
            evaporation_rates,
            [int(m * n_cities) for m in num_ants_multipliers]
        ))

        configs = []
        for alpha, beta, evaporation_rate, num_ants in param_combinations:
            for trial in range(n_trials):
                configs.append({
                    'alpha': alpha,
                    'beta': beta,
                    'evaporation_rate': evaporation_rate,
                    'num_ants': num_ants,
                    'trial': trial,
                    'num_iterations': num_iterations
                })

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [
                executor.submit(self._run_single_trial, config)
                for config in configs
            ]

            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())

        return pd.DataFrame(results)

    def _run_single_trial(self, config):
        aco = AntColonyOptimizer(
            self.distances,
            num_ants=config['num_ants'],
            alpha=config['alpha'],
            beta=config['beta'],
            evaporation_rate=config['evaporation_rate'],
            num_iterations=config["num_iterations"],
            patience=100)

        best_route, best_distance, _, _ = aco.optimize()

        return {
            'alpha': config['alpha'],
            'beta': config['beta'],
            'evaporation_rate': config['evaporation_rate'],
            'num_ants': config['num_ants'],
            'trial': config['trial'],
            'best_distance': best_distance,
        }


@dataclass(frozen=True)
class ACOExperimentVisualizer:
    @staticmethod
    def plot_parameter_effects(results: pd.DataFrame):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Alpha effect
        sns.boxplot(
            data=results,
            x='alpha',
            y='best_distance',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Effect of Alpha')
        axes[0, 0].set_xlabel('Alpha (Pheromone Importance)')
        axes[0, 0].set_ylabel('Solution Length')

        # 2. Beta effect
        sns.boxplot(
            data=results,
            x='beta',
            y='best_distance',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Effect of Beta')
        axes[0, 1].set_xlabel('Beta (Heuristic Importance)')
        axes[0, 1].set_ylabel('Solution Length')

        # 3. Rho effect
        sns.boxplot(
            data=results,
            x='evaporation_rate',
            y='best_distance',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Effect of Rho')
        axes[1, 0].set_xlabel('Rho (Evaporation Rate)')
        axes[1, 0].set_ylabel('Solution Length')

        # 4. Number of ants effect
        sns.boxplot(
            data=results,
            x='num_ants',
            y='best_distance',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Effect of Number of Ants')
        axes[1, 1].set_xlabel('Number of Ants')
        axes[1, 1].set_ylabel('Solution Length')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_parameter_rankings(results: pd.DataFrame):
        rankings = results.groupby(
            ['alpha', 'beta', 'evaporation_rate', 'num_ants']
        )['best_distance'].agg(['mean', 'std']).reset_index()

        # Sort by mean performance
        rankings = rankings.sort_values('mean')

        print("\nTop 5 Parameter Combinations:")
        print("=" * 50)
        for _, row in rankings.head().iterrows():
            print(f"\nParameters:")
            print(f"  Alpha: {row['alpha']}")
            print(f"  Beta: {row['beta']}")
            print(f"  Rho: {row['evaporation_rate']}")
            print(f"  N_ants: {row['num_ants']}")
            print(f"Performance:")
            print(f"  Mean Length: {row['mean']:.2f}")
            print(f"  Std Dev: {row['std']:.2f}")

        print("\nWorst 5 Parameter Combinations:")
        print("=" * 50)
        for _, row in rankings.tail().iterrows():
            print(f"\nParameters:")
            print(f"  Alpha: {row['alpha']}")
            print(f"  Beta: {row['beta']}")
            print(f"  Rho: {row['evaporation_rate']}")
            print(f"  N_ants: {row['num_ants']}")
            print(f"Performance:")
            print(f"  Mean Length: {row['mean']:.2f}")
            print(f"  Std Dev: {row['std']:.2f}")

    @staticmethod
    def analyze_convergence_behavior(results: pd.DataFrame):
        # Calculate coefficient of variation for each parameter combination
        cv_analysis = results.groupby(
            ['alpha', 'beta', 'evaporation_rate', 'num_ants']
        )['best_distance'].agg(['mean', 'std']).reset_index()

        cv_analysis['cv'] = cv_analysis['std'] / cv_analysis['mean']

        # Sort by CV to identify most unstable combinations
        cv_analysis = cv_analysis.sort_values('cv', ascending=False)

        print("\nParameter Combinations Most Likely to Cause Non-convergence:")
        print("=" * 60)
        print("\nBased on Coefficient of Variation (higher = more unstable)")
        for _, row in cv_analysis.head().iterrows():
            print(f"\nParameters:")
            print(f"  Alpha: {row['alpha']}")
            print(f"  Beta: {row['beta']}")
            print(f"  Rho: {row['evaporation_rate']}")
            print(f"  N_ants: {row['num_ants']}")
            print(f"Analysis:")
            print(f"  CV: {row['cv']:.3f}")
            print(f"  Mean Length: {row['mean']:.2f}")
            print(f"  Std Dev: {row['std']:.2f}")

        # Identify potential theoretical issues
        print("\nTheoretical Analysis of Non-convergence Conditions:")

        # Check for exploration/exploitation imbalance
        exploration_issues = cv_analysis[
            (cv_analysis['alpha'] > 2 * cv_analysis['beta']) |
            (cv_analysis['beta'] > 2 * cv_analysis['alpha'])
            ]

        if not exploration_issues.empty:
            print("\nImbalanced Exploration/Exploitation:")
            print("High risk when alpha >> beta (over-exploitation) or")
            print("beta >> alpha (over-exploration)")

        # Check for rapid pheromone decay
        evaporation_issues = cv_analysis[cv_analysis['evaporation_rate'] > 0.7]
        if not evaporation_issues.empty:
            print("\nRapid Pheromone Evaporation:")
            print("High risk with evaporation_rate > 0.7 causing instability")