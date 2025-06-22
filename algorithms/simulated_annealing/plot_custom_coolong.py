import numpy as np
from matplotlib import pyplot as plt


def plot_custom_cooling(iterations_list, temps, T0):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(iterations_list, temps, 'b-', linewidth=2, label='Custom Cooling Schedule')
    plt.title('Custom Cooling Schedule: Oscillating Temperature Pattern', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    cooling_component = [T0 / (1 + 0.1 * k) for k in iterations_list]
    oscillating_component = [0.1 * T0 * np.sin(0.1 * k) for k in iterations_list]

    plt.plot(iterations_list, cooling_component, 'r--', linewidth=2, label='Cooling Component: T₀/(1+0.1k)')
    plt.plot(iterations_list, oscillating_component, 'g--', linewidth=2,
             label='Oscillating Component: 0.1×T₀×sin(0.1k)')
    plt.plot(iterations_list, temps, 'b-', linewidth=2, label='Combined Schedule')
    plt.title('Component Analysis of the Cooling Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()