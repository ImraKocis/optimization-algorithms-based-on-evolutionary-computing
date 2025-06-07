import numpy as np

from algorithms.particle_swarm.particle_swarm import ParticleSwarmOptimizer
from utils.objective_functions import rastrigin_objective_function

"""
For 2dim Rastrigin function:
minimum at [0, 0] with value 0.
maximum at [-4.52299366..., 4.52299366...] with value 80.7.
"""

bounds = np.array([[-5.12, 5.12]] * 2)
pso = ParticleSwarmOptimizer(
    objective_func=rastrigin_objective_function,
    bounds=bounds,
    num_particles=25,
    max_iter=350,
    c1=0.1,
    c2=0.1,
    w_max=0.9,
    w_min=0.2,
    convergence_threshold=1e-12,
    patience=100,
    verbose=True
)
best_pos, best_val, history, _ = pso.optimize()
print("Best Position:", best_pos)
print(f"Best Value: {best_val:.6f}")
