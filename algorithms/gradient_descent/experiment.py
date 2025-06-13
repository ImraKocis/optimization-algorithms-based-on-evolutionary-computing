import numpy as np
from matplotlib import pyplot as plt

from algorithms.gradient_descent.gradient_descent import GradientDescent
from utils.objective_functions import quadratic_objective_function


def gd_experiment():
    step_sizes = np.array([0.1, 1e-5, 1e-10, 1e-15])
    learning_rates = np.array([0.25, 0.1, 0.01, 0.005])
    x0 = 0.0

    results = {}

    for h in step_sizes:
        for lr in learning_rates:
            gd = GradientDescent(
                learning_rate=lr,
                max_iterations=1000,
                tolerance=1e-8,
                gradient_step=h,
                verbose=False,
                use_exact_gradient=False
            )
            result = gd.optimize(quadratic_objective_function, x0=x0)
            key = f"num_h={h}_lr={lr}"
            results[key] = {
                'x_history': result.history.x_values,
                'f_history': result.history.f_values,
                'converged': result.converged,
                'iterations': result.iterations,
                'final_error': abs(result.x_optimal - 2.0)
            }
            # print(f"  {key}: converged={result.converged}, final_error={abs(result.x_optimal - 2.0):.2e}")

    for lr in learning_rates:
        optimizer = GradientDescent(
            learning_rate=lr,
            max_iterations=100,
            tolerance=1e-8,
            verbose=False,
            use_exact_gradient=True
        )
        result = optimizer.optimize(quadratic_objective_function, x0=x0)
        key = f"exact_lr={lr}"
        results[key] = {
            'x_history': result.history.x_values,
            'f_history': result.history.f_values,
            'converged': result.converged,
            'iterations': result.iterations,
            'final_error': abs(result.x_optimal - 2.0)
        }
        # print(f"  {key}: converged={result.converged}, final_error={abs(result.x_optimal - 2.0):.2e}")

    return results


def plot_experiment_results(results):
    x_vals = np.linspace(-1, 5, 500)
    f_vals = (x_vals - 2) ** 2 + 1

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Numerical gradient optimization paths
    ax1.plot(x_vals, f_vals, 'k-', linewidth=2, label='f(x) = (x-2)² + 1')
    ax1.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='True minimum')

    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'red', 'lime', 'magenta',
              'yellow', 'cyan', 'black', 'navy']
    i = 0
    for key, data in results.items():
        if key.startswith('num_h='):
            if len(data["x_history"]) > 2 and len(data["f_history"]) > 2:
                print(f"Plotting {key} with {len(data['x_history'])} points")
                parts = key.split('_')
                h_val = float(parts[1].split('=')[1])
                lr_val = float(parts[2].split('=')[1])

                ax1.plot(data['x_history'], data['f_history'],
                         color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)], markersize=2,
                         label=f'Central Diff: h={h_val:.0e}, lr={lr_val}', alpha=0.8)
            i += 1

    ax1.set_title('Numerical Gradient: Optimization Paths')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend(fontsize='small')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.9, 6)

    # Plot 2: Exact gradient optimization paths
    ax2.plot(x_vals, f_vals, 'k-', linewidth=2, label='f(x) = (x-2)² + 1')
    ax2.axvline(x=2, color='red', linestyle='--', alpha=0.7, label='True minimum')

    i = 0
    for key, data in results.items():
        if key.startswith('exact_lr='):
            if len(data['x_history']) > 2 and len(data['f_history']) > 2:
                lr_val = float(key.split('=')[1])
                ax2.plot(data['x_history'], data['f_history'],
                         color=colors[i % len(colors)], marker='s', markersize=2,
                         label=f'Exact Gradient: lr={lr_val}', alpha=0.8)
            i += 1

    ax2.set_title('Exact Gradient: Optimization Paths')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.9, 6)

    # Plot 3: Numerical gradient convergence (log scale)
    i = 0
    for key, data in results.items():
        if key.startswith('num_h='):
            if len(data['f_history']) > 2:
                parts = key.split('_')
                h_val = float(parts[1].split('=')[1])
                lr_val = float(parts[2].split('=')[1])

                f_diff = np.abs(np.array(data['f_history']) - 1)
                f_diff = np.maximum(f_diff, 1e-16)  # Avoid log(0)
                ax3.semilogy(f_diff[:20], color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)],
                             label=f'Central Diff: h={h_val:.0e}, lr={lr_val}',
                             alpha=0.8, linewidth=2)
            i += 1

    ax3.set_title('Numerical Gradient: Convergence')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('|f(x) - f*| (log scale)')
    ax3.legend(fontsize='small')
    ax3.grid(True, alpha=0.3)

    i = 0
    for key, data in results.items():
        if key.startswith('exact_lr='):
            if len(data['f_history']) > 2:
                lr_val = float(key.split('=')[1])
                f_diff = np.abs(np.array(data['f_history']) - 1)
                f_diff = np.maximum(f_diff, 1e-16)  # Avoid log(0)
                ax4.semilogy(f_diff, color=colors[i % len(colors)],
                             label=f'Exact Gradient: lr={lr_val}',
                             alpha=0.8, linewidth=2)
            i += 1

    ax4.set_title('Exact Gradient Method: Convergence')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('|f(x) - f*| (log scale)')
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
