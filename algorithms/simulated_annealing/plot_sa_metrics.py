import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_sa_results(results):
    plot_data = []
    for schedule_name, metrics_list in results.items():
        for metrics in metrics_list:
            plot_data.append({
                'cooling_schedule': schedule_name,
                'best_value': metrics['best_value'],
                'convergence_iter': metrics['convergence_iter'] if metrics['convergence_iter'] is not None else metrics["n_iter"],
                'acceptance_rate_worse': metrics['acceptance_rate_worse'],
            })

    metrics_df = pd.DataFrame(plot_data)

    table = metrics_df.groupby('cooling_schedule').agg({
         'best_value': ['mean', 'std'],
         'convergence_iter': ['mean', 'std'],
         'acceptance_rate_worse': ['mean', 'std'],
    })

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Boxplot for final best value
    sns.boxplot(x='cooling_schedule', y='best_value', data=metrics_df, ax=axes[0])
    axes[0].set_title('Final best value by cooling schedule')
    axes[0].tick_params(axis="x", rotation=45)

    # Boxplot for convergence iteration
    sns.boxplot(x='cooling_schedule', y='convergence_iter', data=metrics_df, ax=axes[1])
    axes[1].set_title('Convergence iteration by cooling schedule')
    axes[1].tick_params(axis="x", rotation=45)

    # Boxplot for mean acceptance rate of worse solutions
    sns.boxplot(x='cooling_schedule', y='acceptance_rate_worse', data=metrics_df, ax=axes[2])
    axes[2].set_title('Acceptance rate of worse solutions by cooling schedule')
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    print(table)
