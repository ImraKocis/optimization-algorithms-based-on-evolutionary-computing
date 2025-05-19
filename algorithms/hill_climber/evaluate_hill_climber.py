import numpy as np

from algorithms.hill_climber.hill_climber import hill_climber
from algorithms.hill_climber.plot_hill_climber_data import plot_hill_climber_data
from utils.objective_functions import rastrigin_objective_function
from utils.time_complexity_table import create_time_complexity_table


def evaluate_time_complexity_hill_climber(base_max_iterations=100, candidates=50, iterations=10, dimensions=2):
    results = []
    bounds = np.array([[-5.12, 5.12]] * dimensions)
    for i in range(1, iterations + 1):
        max_iterations = base_max_iterations * i
        candidates_per_iteration = candidates * i

        best_sol, best_val, history, result_row = hill_climber(
            rastrigin_objective_function,
            dimensions=dimensions,
            bounds=bounds,
            n_iterations=max_iterations,
            step_size=0.1,
            candidates_per_iteration=candidates_per_iteration,
        )
        results.append(result_row)

    df = create_time_complexity_table(results)
    plot_hill_climber_data(results)
    print("\nTime Complexity Table:")
    return df
