from algorithms.gradient_descent.experiment import gd_experiment, plot_experiment_results

import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


if __name__ == "__main__":
    results = gd_experiment()
    plot_experiment_results(results)

