import numpy as np
import pandas as pd

from algorithms.particle_swarm.evaluate_pso import evaluate_pso
from algorithms.particle_swarm.pso_as_hill_climber_plots import plot_pso_comparison
from utils.objective_functions import rastrigin_objective_function

normal_params = {
    'c1': 2.0,
    'c2': 2.0,
    'w_max': 0.9,
    'w_min': 0.2,
    'num_particles': 40,
    'max_iter': 1500,
    'n_runs': 5,  # More runs for better comparison
    'patience': 150,
    'verbose': False,
    'evaluate_verbose': True
}


pso_as_hill_climber_params = normal_params.copy()
pso_as_hill_climber_params.update({'c1': 0.0, 'c2': 1.0})

bounds = np.array([[-5.12, 5.12]] * 5)

# Evaluate both configurations
print("Evaluating Normal PSO (c1=2.0, c2=2.0)...")
results_normal = evaluate_pso(
    rastrigin_objective_function,
    bounds,
    **normal_params
)

print("\nEvaluating Hill climber PSO (c1=0.0, c2=1.0)...")
results_special = evaluate_pso(
    rastrigin_objective_function,
    bounds,
    **pso_as_hill_climber_params
)

# Combine results into one DataFrame
all_results = results_normal + results_special
results_df = pd.DataFrame(all_results)

# Add configuration labels
results_df['config'] = results_df.apply(
    lambda row: 'Normal PSO (c1=2.0, c2=2.0)' if row['c1'] == 2.0 else 'Hill climber PSO (c1=0.0, c2=1.0)',
    axis=1
)

plot_pso_comparison(results_df)
