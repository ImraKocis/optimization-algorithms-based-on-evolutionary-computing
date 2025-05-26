import numpy as np


def rastrigin_objective_function(x):
    """
    Parameters:
    x (list/array): Input vector (any dimension)
    a (float): Function parameter (default: 10)

    Returns:
    float: Function value
    """
    n = len(x)
    a = 10
    return a * n + sum(x_i ** 2 - a * np.cos(2 * np.pi * x_i) for x_i in x)


def rastrigin_2d(x1, x2, a=10):
    """2D version for visualization"""
    return a * 2 + (x1**2 + x2**2) - a * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))