import matplotlib.pyplot as plt


class Plotter:
    """
    A class to encapsulate plotting methods for optimization progress.

    This class provides:
      - A line plot method (for objective value history over iterations).
      - A 2D path plot method for visualizing the algorithm's movement if solving a 2D problem.
    """

    def __init__(self, title="Optimization Progress", xlabel="Iteration", ylabel="Objective Value"):
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot_line(self, history, label='Best Objective Value'):
        """
        Plots the objective value history as a line plot.

        Parameters:
          - history: A list (or array) of objective values over iterations.
          - label: Label for the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(history, label=label, color='b', marker='o', markersize=3)
        plt.title(self.title)
        plt.xlabel("Iteration")
        plt.ylabel(self.ylabel)
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_path(accepted_positions, best_solution=None):
        """
        Plots the algorithm's movement through the search space, only if the positions are 2D.

        Parameters:
          - accepted_positions: A list of tuples (iteration, position, objective_value).
          - best_solution: The best solution (2D coordinate) to highlight.

        If the dimension of the positions is not 2, the method will notify the user and not plot.
        """
        if not accepted_positions:
            print("No accepted positions to plot.")
            return

        # Check if the position is 2D.
        sample_position = accepted_positions[0][1]
        if sample_position.shape[0] != 2:
            print("The problem is not 2D; cannot plot a 2D movement path.")
            return

        # Extract X and Y coordinates.
        xs = [pos[1][0] for pos in accepted_positions]
        ys = [pos[1][1] for pos in accepted_positions]

        plt.figure(figsize=(10, 6))
        # Plot the path (line connecting the positions).
        plt.plot(xs, ys, linestyle='-', marker='o', color='b', label='Path')
        # Mark the accepted positions with scatter points.
        plt.scatter(xs, ys, color='r', s=50, zorder=5, label='Accepted Positions')

        # Highlight the best solution if provided.
        if best_solution is not None:
            plt.scatter(best_solution[0], best_solution[1], color='g', marker='*', s=200, zorder=10,
                        label='Best Solution')

        plt.title("Algorithm Movement in 2D Space")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.grid(True)
        plt.show()