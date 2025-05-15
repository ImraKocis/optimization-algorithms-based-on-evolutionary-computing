import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.objective_functions import rastrigin_2d


def plot_rastrigin():
    x = np.linspace(-5.12, 5.12, 400)
    y = np.linspace(-5.12, 5.12, 400)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_2d(X, Y)

    # 3D Surface Plot
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X,Y)')
    plt.colorbar(surf)
    plt.title('Rastrigin Function')
    plt.show()

    # Contour Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, 50, cmap='viridis')
    plt.colorbar()
    plt.scatter(0, 0, c='red', label='Global Minimum (0,0)')
    plt.title('Contour Plot of 2D Rastrigin Function')
    plt.legend()
    plt.show()