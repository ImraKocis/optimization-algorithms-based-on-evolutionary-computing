import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg') # remove for jupiter


def plot_performance_heatmap(results_df, metric='best_value'):
    """Create heatmap showing average performance across c1/c2 combinations."""
    # Group by c1, c2 and calculate mean performance
    pivot_data = results_df.groupby(['c1', 'c2'])[metric].mean().unstack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data,
                annot=True,
                fmt='.4f',
                cmap='viridis_r',  # Reverse so darker = better (lower values)
                cbar_kws={'label': f'Average {metric}'})
    plt.title(f'PSO Performance Heatmap: Average {metric}')
    plt.xlabel('c2 (Social Component)')
    plt.ylabel('c1 (Cognitive Component)')
    plt.tight_layout()
    plt.show()


def plot_convergence_heatmap(results_df):
    """Show how quickly different parameter combinations converge."""
    pivot_data = results_df.groupby(['c1', 'c2'])['convergence_iter'].mean().unstack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data,
                annot=True,
                fmt='.1f',
                cmap='plasma_r',
                cbar_kws={'label': 'Average Convergence Iteration'})
    plt.title('PSO Convergence Speed Heatmap')
    plt.xlabel('c2 (Social Component)')
    plt.ylabel('c1 (Cognitive Component)')
    plt.tight_layout()
    plt.show()


def plot_convergence_curves(results_df, top_n=5):
    """Plot convergence curves for best performing parameter combinations."""
    # Find top N parameter combinations
    param_performance = results_df.groupby(['c1', 'c2'])['best_value'].mean().sort_values()
    top_params = param_performance.head(top_n).index

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, top_n))

    for i, (c1, c2) in enumerate(top_params):
        # Get all runs for this parameter combination
        subset = results_df[(results_df['c1'] == c1) & (results_df['c2'] == c2)]

        # Calculate average history across runs
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

        plt.plot(avg_history,
                 color=colors[i],
                 label=f'c1={c1}, c2={c2} (avg: {param_performance.iloc[i]:.4f})',
                 linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Best Value Found')
    plt.title('Convergence Curves: Top Parameter Combinations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for optimization curves
    plt.tight_layout()
    plt.show()


def plot_success_rate_heatmap(results_df, tolerance=1e-5, global_minimum=0.0):
    """Show success rate (% of runs reaching global optimum) for each parameter combo."""

    def success_rate(group):
        successful_runs = (group['best_value'] - global_minimum).abs() < tolerance
        return successful_runs.mean() * 100

    pivot_data = results_df.groupby(['c1', 'c2']).apply(success_rate).unstack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data,
                annot=True,
                fmt='.1f',
                cmap='RdYlGn',  # Red-Yellow-Green for success rates
                cbar_kws={'label': 'Success Rate (%)'})
    plt.title('PSO Success Rate Heatmap (% runs reaching global optimum)')
    plt.xlabel('c2 (Social Component)')
    plt.ylabel('c1 (Cognitive Component)')
    plt.tight_layout()
    plt.show()


def plot_parameter_dashboard(results_df):
    """Create a comprehensive dashboard of parameter effects."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Average performance
    perf_pivot = results_df.groupby(['c1', 'c2'])['best_value'].mean().unstack()
    sns.heatmap(perf_pivot, annot=True, fmt='.4f', ax=axes[0, 0], cmap='viridis_r')
    axes[0, 0].set_title('Average Best Value')

    # 2. Convergence speed
    conv_pivot = results_df.groupby(['c1', 'c2'])['convergence_iter'].mean().unstack()
    sns.heatmap(conv_pivot, annot=True, fmt='.1f', ax=axes[0, 1], cmap='plasma_r')
    axes[0, 1].set_title('Average Convergence Iteration')

    # 3. Standard deviation (consistency)
    std_pivot = results_df.groupby(['c1', 'c2'])['best_value'].std().unstack()
    sns.heatmap(std_pivot, annot=True, fmt='.4f', ax=axes[1, 0], cmap='Reds')
    axes[1, 0].set_title('Performance Standard Deviation')

    # 4. Success rate
    def success_rate(group):
        return ((group['best_value'] - 0.0).abs() < 1e-5).mean() * 100

    success_pivot = results_df.groupby(['c1', 'c2']).apply(success_rate).unstack()
    sns.heatmap(success_pivot, annot=True, fmt='.1f', ax=axes[1, 1], cmap='RdYlGn')
    axes[1, 1].set_title('Success Rate (%)')

    plt.tight_layout()
    plt.show()