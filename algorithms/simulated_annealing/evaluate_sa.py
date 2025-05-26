from algorithms.simulated_annealing.simulated_annealing import simulated_annealing


def evaluate_sa(
    objective_func,
    bounds,
    global_minimum,
    n_iterations,
    step_size,
    T0,
    cooling_schedule="linear",
    cooling_kwargs=None,
    tolerance=1e-3,
    verbose=False,
    n_trials=10
):
    all_metrics = []
    for trial in range(n_trials):
        best_solution, best_value, history, acceptance_rate, acceptance_rate_worse = simulated_annealing(
            objective_func,
            bounds,
            n_iterations=n_iterations,
            step_size=step_size,
            T0=T0,
            cooling_schedule=cooling_schedule,
            cooling_kwargs=cooling_kwargs,
            verbose=verbose,
            min_temperature=1e-6
        )

        convergence_iter = None
        for idx, val in enumerate(history):
            if abs(val - global_minimum) < tolerance:
                convergence_iter = idx
                break

        metrics = {
            "best_value": best_value,
            "convergence_iter": convergence_iter,
            "history": history,
            "best_solution": best_solution,
            "acceptance_rate_worse": acceptance_rate_worse,
            "n_iter": n_iterations,
        }
        all_metrics.append(metrics)

    return all_metrics
