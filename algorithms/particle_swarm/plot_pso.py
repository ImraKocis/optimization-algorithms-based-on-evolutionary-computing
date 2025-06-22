import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_parameter_dashboard(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Average performance

    # Groups results by c1 and c2 parameter combinations
    # Calculates the average best_value for each combination across all runs
    # Reshapes data using .unstack() to create a matrix where rows=c1, columns=c2
    perf_pivot = results_df.groupby(['c1', 'c2'])['best_value'].mean().unstack()
    sns.heatmap(perf_pivot, annot=True, fmt='.4f', ax=axes[0, 0], cmap='viridis_r')
    axes[0, 0].set_title('Average Best Value')

    # 2. Convergence speed

    # Groups results by c1 and c2 parameter combinations
    # Calculates the average convergence_iter for each combination across all runs
    # Reshapes data using .unstack() to create a matrix where rows=c1, columns=c2
    conv_pivot = results_df.groupby(['c1', 'c2'])['convergence_iter'].mean().unstack()
    sns.heatmap(conv_pivot, annot=True, fmt='.1f', ax=axes[0, 1], cmap='plasma_r')
    axes[0, 1].set_title('Average Convergence Iteration')

    def success_rate(group):
        return ((group['best_value'] - 0.0).abs() < 1e-5).mean() * 100

    success_pivot = results_df.groupby(['c1', 'c2']).apply(success_rate).unstack()
    sns.heatmap(success_pivot, annot=True, fmt='.1f', ax=axes[1, 0], cmap='RdYlGn')
    axes[1, 0].set_title('Success Rate (%)')

    plt.tight_layout()
    plt.show()


def plot_weight_impact_analysis(results_df):
    pivot_data = results_df.groupby(['w_max', 'c1', 'c2'])['best_value'].mean().reset_index()
    pivot_data['param_combo'] = pivot_data['c1'].astype(str) + ',' + pivot_data['c2'].astype(str)
    heatmap_data = pivot_data.pivot(index='param_combo', columns='w_max', values='best_value')

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis_r',
                cbar_kws={'label': 'Average Best Value'})
    plt.title('Weight Impact Across c1,c2 Combinations')
    plt.xlabel('w_max (Maximum Inertia Weight)')
    plt.ylabel('c1,c2 Parameter Combination')
    plt.tight_layout()
    plt.show()


def plot_convergence_curves(results_df, top_n=10):
    results_df['param_label'] = results_df.apply(
        lambda row: f"c1={row['c1']}, c2={row['c2']}, w_max={row['w_max']}", axis=1
    )

    # find top N parameter combinations by average performance
    param_performance = results_df.groupby('param_label')['best_value'].mean().sort_values()
    top_params = param_performance.head(top_n).index.tolist()

    plt.figure(figsize=(12, 8))
    colors = sns.color_palette('tab10', n_colors=len(top_params))

    for i, param in enumerate(top_params):
        # Get all runs for this parameter combination
        subset = results_df[results_df['param_label'] == param]

        # Calculate average history across all runs
        max_len = max(len(hist) for hist in subset['history'])
        avg_history = []

        for iter_idx in range(max_len):
            iter_values = []
            for hist in subset['history']:
                if iter_idx < len(hist):
                    iter_values.append(hist[iter_idx])
                else:
                    iter_values.append(hist[-1])  # Use last value if run ended early
            avg_history.append(np.mean(iter_values))

        avg_final = param_performance[param]
        plt.plot(avg_history,
                 color=colors[i],
                 label=f'{param} (avg: {avg_final:.4f})',
                 linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Average Best Value')
    plt.title(f'Top {top_n} Solution Over Iterations for Different Parameter Settings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.yscale('log')  # Log scale for better visualization of Rastrigin convergence
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
