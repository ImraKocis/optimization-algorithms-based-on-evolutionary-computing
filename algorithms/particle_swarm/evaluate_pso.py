import time

from algorithms.particle_swarm.particle_swarm import ParticleSwarmOptimizer


def evaluate_pso(
        objective_func,
        bounds,
        c1,
        c2,
        w_max,
        w_min=0.2,
        num_particles=30,
        max_iter=500,
        n_runs=30,
        tolerance=1e-5,
        global_minimum=0.0
):
    results = []
    for run in range(n_runs):
        pso = ParticleSwarmOptimizer(
            objective_func=objective_func,
            bounds=bounds,
            num_particles=num_particles,
            max_iter=max_iter,
            w_max=w_max,
            w_min=w_min,
            c1=c1,
            c2=c2,
            patience=max_iter # patience is set to max_iter for evaluation purposes
        )
        start_time = time.time()
        best_pos, best_val, history = pso.optimize()
        elapsed = time.time() - start_time

        convergence_iter = None
        for idx, val in enumerate(history):
            if abs(val - global_minimum) < tolerance:
                convergence_iter = idx
                break

        results.append({
            'c1': c1,
            'c2': c2,
            'w_max': w_max,
            'w_min': w_min,
            'run': run + 1,
            'best_value': best_val,
            'convergence_iter': convergence_iter if convergence_iter is not None else max_iter,
            'time': elapsed,
            'history': history
        })
    return results


