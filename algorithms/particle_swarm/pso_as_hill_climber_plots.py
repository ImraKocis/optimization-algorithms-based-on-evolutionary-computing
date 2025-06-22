import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pso_comparison(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    sns.boxplot(data=results_df, x='config', y='best_value', ax=axes[0, 0])
    axes[0, 0].set_title('Performance Comparison: Final Best Values')
    axes[0, 0].set_ylabel('Best Value Found')
    axes[0, 0].tick_params(axis='x')
    axes[0, 0].grid(True, alpha=0.3)

    sns.boxplot(data=results_df, x='config', y='convergence_iter', ax=axes[0, 1])
    axes[0, 1].set_title('Convergence Speed Comparison')
    axes[0, 1].set_ylabel('Iterations to Convergence')
    axes[0, 1].tick_params(axis='x')
    axes[0, 1].grid(True, alpha=0.3)

    configs = results_df['config'].unique()
    colors = ['blue', 'red']

    for i, config in enumerate(configs):
        subset = results_df[results_df['config'] == config]

        max_len = max(len(hist) for hist in subset['history'])
        avg_history = []

        for iter_idx in range(max_len):
            iter_values = []
            for hist in subset['history']:
                if iter_idx < len(hist):
                    iter_values.append(hist[iter_idx])
                else:
                    iter_values.append(hist[-1])
            avg_history.append(np.mean(iter_values))

        axes[1, 0].plot(avg_history, color=colors[i], label=config, linewidth=2)

    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Average Best Value')
    axes[1, 0].set_title('Average Convergence Curves')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
