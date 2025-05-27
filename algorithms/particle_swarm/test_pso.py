import numpy as np

from algorithms.particle_swarm.particle_swarm import ParticleSwarmOptimizer
from utils.objective_functions import rastrigin_objective_function

bounds = np.array([[-5.12, 5.12]] * 2)
pso = ParticleSwarmOptimizer(
    objective_func=rastrigin_objective_function,
    bounds=bounds,
    num_particles=30,
    max_iter=1000,
    w=0.9,
    c1=2,
    c2=2,
    linear_weight_decay=True,
    linear_weight_decay_alpha=0.0008
)
best_pos, best_val, history = pso.optimize(verbose=True)
print("Best Position:", best_pos)
print("Best Value:", best_val)
