from algorithms.simulated_annealing.cooling_schedules import custom_cooling
from algorithms.simulated_annealing.plot_custom_coolong import plot_custom_cooling

import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


if __name__ == "__main__":
    T0 = 100
    iterations = 100

    iterations_list = list(range(iterations))
    temps = [custom_cooling(None, k, T0) for k in iterations_list]

    plot_custom_cooling(iterations_list, temps, T0)


