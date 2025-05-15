import numpy as np


def hill_climber(objective, x0, num_iterations=10000, step_size=0.1):
    """
    A general hill-climbing algorithm that works in n-dimensional space.

    Parameters:
      - objective: The objective function to minimize.
      - x0: The initial solution (a numpy array of any dimension).
      - num_iterations: Total number of iterations to perform.
      - step_size: Maximum perturbation added to each dimension.

    Returns:
      - best_solution: The best solution found.
      - best_value: The objective function value of the best solution.
      - best_values_history: A list of the best objective values over iterations.
      - accepted_positions: A list of tuples (iteration, position, objective_value) for each accepted move.
    """
    # Initialize the current solution and its objective value.
    current_solution = x0.copy()
    current_value = objective(current_solution)

    # Best solution so far.
    best_solution = current_solution.copy()
    best_value = current_value

    # History: best objective values and accepted moves.
    best_values_history = [best_value]
    accepted_positions = [(0, current_solution.copy(), current_value)]

    for i in range(num_iterations):
        # Generate a candidate solution by perturbing the current solution.
        candidate = current_solution + np.random.uniform(-step_size, step_size, size=x0.shape)
        candidate_value = objective(candidate)

        # Accept the candidate if it improves the objective value.
        if candidate_value < current_value:
            current_solution = candidate.copy()
            current_value = candidate_value
            accepted_positions.append((i + 1, current_solution.copy(), current_value))
            # Update the best solution if this candidate is the best so far.
            if candidate_value < best_value:
                best_solution = candidate.copy()
                best_value = candidate_value

        best_values_history.append(best_value)

        # Optionally print progress every 1000 iterations.
        if (i + 1) % 1000 == 0:
            print(f"Iteration {i + 1}: Best Value = {best_value:.6f}")

    return best_solution, best_value, best_values_history, accepted_positions
