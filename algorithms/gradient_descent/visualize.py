import numpy as np
from matplotlib import pyplot as plt


def gradient_descent_visualize_optimization(f, history, title="Gradient Descent Optimization"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Function and optimization path
    x_range = np.linspace(-1, 5, 1000)
    y_range = [f(x) for x in x_range]

    ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x)')
    ax1.plot(history.x_values, history.f_values, 'ro-',
             markersize=4, linewidth=1, label='Optimization path')
    ax1.plot(history.x_values[0], history.f_values[0], 'go',
             markersize=8, label='Start')
    ax1.plot(history.x_values[-1], history.f_values[-1], 'rs',
             markersize=8, label='End')

    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Function and Optimization Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Convergence analysis
    ax2.semilogy(history.gradient_norms, 'b-', linewidth=2)  # Use semilogy for better visibility of low values
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm (log scale)')
    ax2.set_title('Convergence: Gradient Norm vs Iteration')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
