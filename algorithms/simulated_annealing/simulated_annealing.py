import numpy as np

from algorithms.simulated_annealing.cooling_schedules import linear_cooling, COOLING_SCHEDULES


def simulated_annealing(
    objective_func,
    bounds,
    n_iterations=1000,
    step_size=0.1,
    T0=1.0,
    cooling_schedule="linear",
    cooling_kwargs=None,
    verbose=False,
    min_temperature=1e-8
):
    """
    Simulated Annealing algorithm for optimization.
    :param objective_func:
    :param bounds: for objective function, shape np.array([[min, max]] * dimensions)
    :param n_iterations: num of iterations to run
    :param step_size
    :param T0: initial temp
    :param cooling_schedule: one of "linear", "logarithmic", "exponential", "adaptive", "custom"
    :param cooling_kwargs: additional parameters for cooling function, e.g. {'acceptance_rate': 0.44, 'adapt_factor': 0.95, 'window': 50, 'last_accepts': 0}
    :param verbose: if True, prints detailed info about each iteration
    :param min_temperature: minimum temperature to avoid division by zero or log(0)
    :return: best_solution, best_value, history, acceptance_rate, acceptance_rate_worse
    """
    dimensions = bounds.shape[0]
    solution = np.random.uniform(bounds[:, 0], bounds[:, 1], size=dimensions)
    value = objective_func(solution)
    best_solution = solution.copy()
    best_value = value
    history = [best_value]
    T = T0
    accepted = 0
    total_accepted = 0
    accepted_worse = 0

    cooling_func = COOLING_SCHEDULES.get(cooling_schedule, linear_cooling)
    if cooling_kwargs is None:
        cooling_kwargs = {}

    window = cooling_kwargs.get('window', 50)

    for k in range(n_iterations):
        if T < min_temperature:
            break

        candidate = solution + np.random.randn(dimensions) * step_size
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        candidate_value = objective_func(candidate)
        delta = candidate_value - value

        if delta < 0:
            accept = True
        else:
            exponent = -delta / T
            exponent = np.clip(exponent, -700, 700)
            accept = np.random.random() < np.exp(exponent)
        if accept:
            solution = candidate
            value = candidate_value
            accepted += 1
            total_accepted += 1
            if delta > 0:
                accepted_worse += 1
            if value < best_value:
                best_solution = solution.copy()
                best_value = value

        history.append(best_value)

        if cooling_schedule == "adaptive":
            if (k + 1) % window == 0 and k+1 != n_iterations:
                current_acceptance_rate = accepted / window
                T = cooling_func(T, k, T0, current_acceptance_rate=current_acceptance_rate, verbose=verbose, **cooling_kwargs)
                if verbose:
                    print(f"Iter {k}, T={T:.4f}, accepted in window={accepted}, best_value={best_value:.6f}")
                accepted = 0
            else:
                pass
        elif cooling_schedule == "logarithmic":
            T = cooling_func(T, k, T0=T0, verbose=verbose)
        elif cooling_schedule == "exponential":
            cooling_kwargs_filtered = {key: val for key, val in cooling_kwargs.items() if
                                       key in ['beta', 'verbose']}
            T = cooling_func(T, k, T0=T0, **cooling_kwargs_filtered)
        else:
            cooling_kwargs_filtered = {key: val for key, val in cooling_kwargs.items() if
                                       key not in ['window', 'last_accepts', 'acceptance_rate', 'adapt_factor']}
            T = cooling_func(T, k, T0=T0, **cooling_kwargs_filtered)

        if verbose and cooling_schedule != "adaptive" and (k % (n_iterations // 10) == 0):
            print(f"Iter {k}, T={T:.4f}, best_value={best_value:.6f}")

    acceptance_rate = total_accepted / n_iterations if n_iterations > 0 else 0
    acceptance_rate_worse = accepted_worse / n_iterations if n_iterations > 0 else 0

    return best_solution, best_value, history, acceptance_rate, acceptance_rate_worse
