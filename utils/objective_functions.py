import numpy as np


def rastrigin_objective_function(x):
    n = len(x)
    A = 10
    return A * n + sum(x_i ** 2 - A * np.cos(2 * np.pi * x_i) for x_i in x)


def rastrigin_2d(x1, x2, a=10):
    return a * 2 + (x1**2 + x2**2) - a * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))


def quadratic_objective_function(x):
    return (x - 2)**2 + 1
