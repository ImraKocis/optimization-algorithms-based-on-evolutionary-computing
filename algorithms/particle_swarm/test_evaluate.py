import numpy as np
import pandas as pd

from algorithms.particle_swarm.evaluate_pso import evaluate_pso
from algorithms.particle_swarm.plot_pso import plot_performance_heatmap, plot_convergence_heatmap, \
    plot_convergence_curves, plot_success_rate_heatmap, plot_parameter_dashboard
from utils.objective_functions import rastrigin_objective_function

c1_values = [1.0, 1.5, 2.0]
c2_values = [1.0, 1.5, 2.0]
w_max_values = [0.5, 0.7, 0.9]
w_min = 0.2
all_results = []
bounds = np.array([[-5.12, 5.12]] * 2)

for c1 in c1_values:
    for c2 in c2_values:
        for w_max in w_max_values:
            print(f"Running for c1={c1}, c2={c2}, w_max={w_max}, w_min={w_min}")
            results = evaluate_pso(
                rastrigin_objective_function,
                bounds,
                c1,
                c2,
                w_max=w_max,
                w_min=w_min,
                num_particles=20,
                max_iter=250,
                n_runs=10,
            )

            all_results.extend(results)

results_df = pd.DataFrame(all_results)

plot_performance_heatmap(results_df)
plot_convergence_heatmap(results_df)
plot_convergence_curves(results_df)
plot_success_rate_heatmap(results_df)
plot_parameter_dashboard(results_df)
