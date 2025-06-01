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
        patience=50,
        verbose=False,
        evaluate_verbose=False
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
            patience=patience,
            verbose=verbose
        )
        start_time = time.time()
        best_pos, best_val, history, convergence_iter = pso.optimize()
        elapsed = time.time() - start_time

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
        if evaluate_verbose:
            print(f"Run {run + 1}/{n_runs}: Best Value = {best_val:.6f}, Time = {elapsed:.2f}s")

    return results


