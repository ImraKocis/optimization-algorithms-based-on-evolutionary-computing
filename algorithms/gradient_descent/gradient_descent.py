import numpy as np

from algorithms.gradient_descent.optimization_history import OptimizationHistory
from algorithms.gradient_descent.optimization_result import OptimizationResult
from utils.objective_functions import quadratic_objective_function


class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=1000,
                 tolerance=1e-6, gradient_step=1e-5, use_exact_gradient=False, verbose=True):
        self.learning_rate = learning_rate  # usually 0.01 but can vary, it's highly problem-dependent
        self.max_iterations = max_iterations  # usually 1000 or more, depends on problem complexity and learning rate
        self.tolerance = tolerance
        self.gradient_step = gradient_step  # noise in data
        self.verbose = verbose
        self.history = OptimizationHistory()
        self.use_exact_gradient = use_exact_gradient

    def central_numerical_gradient(self, objective_func, x):
        """
        For scalar: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        For vector: f'(x_i) ≈ [f(x+h*e_i) - f(x-h*e_i)] / (2h)
        """
        x = np.asarray(x, dtype=float)

        # scalar case
        if x.ndim == 0:
            return (objective_func(x + self.gradient_step) -
                    objective_func(x - self.gradient_step)) / (2 * self.gradient_step)

        # vector case
        gradient = np.zeros_like(x)
        for i in range(len(x)):
            x_forward = x.copy()
            x_backward = x.copy()
            x_forward[i] += self.gradient_step
            x_backward[i] -= self.gradient_step
            gradient[i] = (objective_func(x_forward) -
                           objective_func(x_backward)) / (2 * self.gradient_step)

        return gradient

    @staticmethod
    def exact_gradient_quadratic(x):
        return 2 * (x - 2)

    def gradient(self, objective_func, x):
        if self.use_exact_gradient:
            return self.exact_gradient_quadratic(x)
        else:
            return self.central_numerical_gradient(objective_func, x)

    def optimize(self, objective_func, x0):

        x_current = np.asarray(x0, dtype=float).copy()
        self.history.reset()

        initial_value = objective_func(x_current)
        self.history.record_iteration(x_current, initial_value, np.array([]), 0.0)

        if self.verbose:
            print(f"Starting Gradient Descent from x0 = {x_current}")
            print(f"Initial function value: f(x0) = {initial_value:.6f}")
            print("-" * 50)

        for iteration in range(self.max_iterations):
            gradient = self.gradient(objective_func, x_current)
            grad_norm = np.linalg.norm(gradient)

            if grad_norm < self.tolerance:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                    print(f"Final gradient norm: {grad_norm:.2e}")
                break

            x_current = x_current - self.learning_rate * gradient

            current_value = objective_func(x_current)

            self.history.record_iteration(x_current, current_value, gradient, grad_norm)

            if self.verbose and iteration % 20 == 0:
                print(f"Iteration {iteration:4d}: f(x) = {current_value:8.6f}, ||grad|| = {grad_norm:.2e}")

        else:
            if self.verbose:
                print(f"Maximum iterations ({self.max_iterations}) reached")
                print(f"Final gradient norm: {grad_norm:.2e}")

        return OptimizationResult(
            x_optimal=x_current,
            f_optimal=objective_func(x_current),
            iterations=len(self.history.x_values) - 1,
            converged=grad_norm < self.tolerance,
            history=self.history
        )
