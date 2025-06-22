import numpy as np

from algorithms.particle_swarm.particle_swarm import ParticleSwarmOptimizer
from utils.objective_functions import rastrigin_objective_function

"""
For 2dim Rastrigin function:
minimum at [0, 0] with value 0.
maximum at [-4.52299366..., 4.52299366...] with value 80.7.
"""

bounds = np.array([[-5.12, 5.12]] * 5)
pso = ParticleSwarmOptimizer(
    objective_func=rastrigin_objective_function,
    bounds=bounds,
    num_particles=25,
    max_iter=200,
    c1=2,
    c2=2,
    w_max=0.9,
    w_min=0.2,
    convergence_threshold=1e-12,
    patience=50,
    verbose=True,
    maximize=False
)
best_pos, best_val, history, _ = pso.optimize()
print("Best Position:", best_pos)
print(f"Best Value: {best_val:.6f}")
