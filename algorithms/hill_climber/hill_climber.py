import numpy as np
import time


def hill_climber(objective_func, dimensions, bounds, n_iterations=1000, step_size=0.1, candidates_per_iteration=0):
    """
    Algo info:

    Large step size is fast, but can skip over the global minimum-more "greedy" and less thorough.

    Small step size is slower, but more precise and less likely to miss the global minimum, but it may get stuck
    in a local minimum or take longer to converge.

    Parameters:
        objective_func: Callable, the objective function to minimize.
        dimensions: int, number of dimensions.
        bounds: np.ndarray, shape (dimensions, 2), lower and upper bounds for each variable.
        n_iterations: int, number of iterations.
        step_size: float, standard deviation for random step (Gaussian).
        candidates_per_iteration: int, number of local search steps.

    Returns:
        best_solution: np.ndarray, best found solution.
        best_value: float, objective value at best_solution.
        history: list, best_value at each iteration (for plotting).
    """

    # random starting point within some bounds
    solution = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=dimensions)
    # best solution so far - init solution - placeholder in format [num, num] * dimensions
    best_solution = solution.copy()
    # best value so far - placeholder - single value
    best_value = objective_func(solution)
    # history of best values
    history = [best_value]

    # start tracking main time
    time_start = time.time()
    time_total_local_search = 0.0
    time_max_iter = 0.0

    for i in range(n_iterations):
        # start tracking time for this iteration
        iter_start = time.time()
        # generate a candidate solution by taking a random step in some cases this
        # can result with a solution outside the bounds
        candidate = solution + np.random.randn(dimensions) * step_size
        # ensure candidate is within bounds by setting out-of-bounds values to the nearest bound value
        candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
        # evaluate candidate solution
        candidate_eval = objective_func(candidate)

        # if the candidate is better, move to it
        if candidate_eval < best_value:
            # current solution is this candidate, and we will take next step from this candidate.
            solution = candidate
            # why copy()? If we just wrote best_solution = candidate, then if we later change candidate,
            # best_solution would also change (because both point to the same array in memory).
            best_solution = candidate.copy()
            best_value = candidate_eval

        t_local_start = time.time()
        for _ in range(candidates_per_iteration):
            local_candidate = solution + np.random.randn(dimensions) * (step_size / 2)
            local_candidate = np.clip(local_candidate, bounds[:, 0], bounds[:, 1])
            local_candidate_eval = objective_func(local_candidate)

            if local_candidate_eval < best_value:
                solution = local_candidate
                best_solution = local_candidate.copy()
                best_value = local_candidate_eval
        t_local_end = time.time()
        time_total_local_search += (t_local_end - t_local_start)

        iter_end = time.time()
        iter_time = iter_end - iter_start
        if iter_time > time_max_iter:
            time_max_iter = iter_time

        history.append(best_value)

    time_end = time.time()
    time_total = time_end - time_start

    result_row = {
        'input_size_max_iter': n_iterations,
        'input_size_local_search': candidates_per_iteration,
        'time_max_iter': time_max_iter,
        'time_local_search': time_total_local_search,
        'time_total': time_total
    }

    return best_solution, best_value, history, result_row
