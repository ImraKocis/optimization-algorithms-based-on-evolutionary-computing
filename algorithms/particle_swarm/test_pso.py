import numpy as np

from algorithms.particle_swarm.particle_swarm import ParticleSwarmOptimizer
from utils.objective_functions import rastrigin_objective_function

"""
For 2dim Rastrigin function:
minimum at [0, 0] with value 0.
maximum at [-4.52299366..., 4.52299366...] with value 80.63.
"""

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
    linear_weight_decay_alpha=0.0008,
    algo_type="maximizer"
)
best_pos, best_val, history = pso.optimize(verbose=True)
print("Best Position:", best_pos)
print("Best Value:", best_val)
