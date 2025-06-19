from itertools import product
import numpy as np
import pandas as pd
from algorithms.fa.firefly_algorithm import FireflyAlgorithm
from algorithms.fa.plot import fa_plots, plot_fireflies_positions


def fa_analyze_hyperparameters(objective_function,
                               bounds,
                               dimensions=None,
                               alpha_values=None,
                               beta0_values=None,
                               gamma_values=None,
                               runs_per_config=5,
                               max_generations=200):

    if alpha_values is None:
        alpha_values = np.arange(0.1, 1.1, 0.2)
    if beta0_values is None:
        beta0_values = np.arange(0.1, 1.1, 0.2)
    if gamma_values is None:
        gamma_values = np.arange(0.1, 2.1, 0.3)
    if dimensions is None:
        dimensions = len(bounds)

    print(f"Analyzing {len(alpha_values)} alpha values × {len(beta0_values)} beta0 values × {len(gamma_values)} gamma values")
    print(f"Total configurations: {len(alpha_values) * len(beta0_values) * len(gamma_values)}")
    print(f"Runs per configuration: {runs_per_config}")
    print(f"Total FA runs: {len(alpha_values) * len(beta0_values) * len(gamma_values) * runs_per_config}")

    results = {
        'alpha': [],
        'beta0': [],
        'gamma': [],
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_generation': [],
        'final_positions': [],
        'best_position': []
    }

    config_count = 0
    total_configs = len(alpha_values) * len(beta0_values) * len(gamma_values)

    for alpha, beta0, gamma in product(alpha_values, beta0_values, gamma_values):
        config_count += 1
        run_results = []
        convergence_gens = []
        run_positions = []
        run_best_positions = []

        for run in range(runs_per_config):
            fa = FireflyAlgorithm(
                objective_function=objective_function,
                dimensions=dimensions,
                bounds=bounds,
                population_size=30,
                alpha=alpha,
                beta0=beta0,
                gamma=gamma,
                patience=30,
            )

            best_pos, best_fit = fa.optimize(max_generations=max_generations, verbose=False)

            run_results.append(best_fit)
            convergence_gens.append(len(fa.best_fitness_history))
            run_best_positions.append(best_pos.copy())
            final_positions = np.array([firefly.position.copy() for firefly in fa.swarm])
            run_positions.append(final_positions)

        best_fitness = np.min(run_results)
        mean_fitness = np.mean(run_results)
        std_fitness = np.std(run_results)
        avg_convergence = np.mean(convergence_gens)

        results['alpha'].append(alpha)
        results['beta0'].append(beta0)
        results['gamma'].append(gamma)
        results['best_fitness'].append(best_fitness)
        results['mean_fitness'].append(mean_fitness)
        results['std_fitness'].append(std_fitness)
        results['convergence_generation'].append(avg_convergence)
        results['final_positions'].append(run_positions)
        results['best_position'].append(run_best_positions[np.argmin(run_results)])

        if config_count % max(1, total_configs // 10) == 0:
            print(f"Progress: {config_count}/{total_configs} ({config_count / total_configs * 100:.1f}%) - "
                  f"α={alpha:.1f}, β₀={beta0:.1f}, γ={gamma:.1f} → Best: {best_fitness:.4f}")

    df = pd.DataFrame(results)
    fa_plots(df)
    plot_fireflies_positions(df, bounds)
