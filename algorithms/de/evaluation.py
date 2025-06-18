import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from algorithms.de.differential_evolution import DifferentialEvolution


def de_analyze_hyperparameters(objective_function, bounds,
                            F_values=None, CR_values=None,
                            runs_per_config=5, max_generations=200):
    if F_values is None:
        F_values = np.arange(0.1, 1.1, 0.1)  # 0.1 to 1.0
    if CR_values is None:
        CR_values = np.arange(0.1, 1.0, 0.1)  # 0.1 to 0.9

    print(f"Analyzing {len(F_values)} F values × {len(CR_values)} CR values")
    print(f"Total configurations: {len(F_values) * len(CR_values)}")
    print(f"Runs per configuration: {runs_per_config}")
    print(f"Total DE runs: {len(F_values) * len(CR_values) * runs_per_config}")

    results = {
        'F': [],
        'CR': [],
        'best_fitness': [],
        'final_diversity': [],
    }

    for F, CR in product(F_values, CR_values):
        run_results = []
        diversities = []

        for run in range(runs_per_config):
            de = DifferentialEvolution(
                objective_function=objective_function,
                bounds=bounds,
                population_size=10 * len(bounds),
                patience=50,
                F=F,
                CR=CR,
            )

            result = de.optimize(max_generations=max_generations, verbose=False)

            run_results.append(result['best_fitness'])
            diversities.append(de.diversity_history[-1])

        best_fitness = np.min(run_results)
        avg_diversity = np.mean(diversities)

        results['F'].append(F)
        results['CR'].append(CR)
        results['best_fitness'].append(best_fitness)
        results['final_diversity'].append(avg_diversity)

        print(f"Best: {best_fitness:.4f}")

    return results


def de_plot_parameter_heatmaps(results, F_values, CR_values):
    fitness_matrix = np.array(results['best_fitness']).reshape(len(F_values), len(CR_values))
    diversity_matrix = np.array(results['final_diversity']).reshape(len(F_values), len(CR_values))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(fitness_matrix,
                xticklabels=[f'{cr:.1f}' for cr in CR_values],
                yticklabels=[f'{f:.1f}' for f in F_values],
                annot=True, fmt='.1f', cmap='viridis',
                ax=axes[0])
    axes[0].set_title('Solution Quality')
    axes[0].set_xlabel('CR (Crossover Rate)')
    axes[0].set_ylabel('F (Mutation Factor)')

    sns.heatmap(diversity_matrix,
                xticklabels=[f'{cr:.1f}' for cr in CR_values],
                yticklabels=[f'{f:.1f}' for f in F_values],
                annot=True, fmt='.3f', cmap='plasma',
                ax=axes[1])
    axes[1].set_title('Final Population Diversity')
    axes[1].set_xlabel('CR (Crossover Rate)')
    axes[1].set_ylabel('F (Mutation Factor)')

    plt.tight_layout()
    plt.show()
