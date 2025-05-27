import numpy as np

from algorithms.simulated_annealing.simulated_annealing import simulated_annealing
from utils.objective_functions import rastrigin_objective_function

bounds = np.array([[-5.12, 5.12]] * 2)
best_sol, best_val, history, acceptance_rate, acceptance_rate_worse = simulated_annealing(
        rastrigin_objective_function,
        bounds,
        n_iterations=1000,
        step_size=0.2,
        T0=100,
        cooling_schedule="adaptive",
        cooling_kwargs={'learning_rate': 0.5, 'cooling_rate': 0.95},
        verbose=True
    )
print("Best Solution:", best_sol)
print("Best Value:", best_val)
print("Acceptance_rate", acceptance_rate)
print("Acceptance_rate_worse", acceptance_rate_worse)
