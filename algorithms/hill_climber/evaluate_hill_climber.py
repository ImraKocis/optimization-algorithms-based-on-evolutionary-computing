from algorithms.hill_climber.hill_climber import hill_climber_with_timing


def evaluate_hill_climber_timing(objective, initial_solution, m_values, n_values, step_size=0.1):
    """
    Evaluates the hill climber for various outer iterations (m_values)
    and inner iterations (n_values). Returns a list of dictionaries containing
    the input parameters and timing measurements.

    Each dictionary contains:
      - input_size_max_iter (m)
      - input_size_local_search (n)
      - time_max_iter (sec): total outer loop overhead time
      - time_local_search (sec): total inner loop time
      - time_total (sec): sum of outer and inner times
      - product: m * n (for plotting scaling)
    """
    results = []
    for m in m_values:
        for n in n_values:
            # Run the hill climber and measure times.
            _, _, outer_time, inner_time, total_time = hill_climber_with_timing(
                objective, initial_solution, max_iter=m, local_search=n, step_size=step_size)
            results.append({
                'input_size_max_iter': m,
                'input_size_local_search': n,
                'time_max_iter': outer_time,
                'time_local_search': inner_time,
                'time_total': total_time,
                'product': m * n
            })
    return results
