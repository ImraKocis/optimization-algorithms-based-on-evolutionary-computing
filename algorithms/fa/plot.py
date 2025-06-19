import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def fa_plots(df):
    plt.style.use('default')
    sns.set_palette("husl")

    fig = plt.figure(figsize=(20, 6))

    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # Plot 1: Heatmap (α vs β₀, averaged over γ)
    heatmap_data = df.groupby(['alpha', 'beta0'])['best_fitness'].mean().unstack()
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis_r', ax=ax1)
    ax1.set_title('Performance Heatmap\n(α vs β₀, lower is better)')
    ax1.set_xlabel('Beta0 (β₀)')
    ax1.set_ylabel('Alpha (α)')

    # Plot 2: Parameter Correlation Matrix
    scatter = ax2.scatter(df['convergence_generation'], df['best_fitness'],
                          c=df['alpha'], cmap='plasma', alpha=0.6, s=50)
    ax2.set_xlabel('Average Convergence Generation')
    ax2.set_ylabel('Best Fitness')
    ax2.set_title('Performance vs Convergence Speed\n(colored by α)')
    plt.colorbar(scatter, ax=ax2, label='Alpha (α)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Best vs Worst Comparison
    best_configs = df.nsmallest(10, 'best_fitness')

    config_labels = [f'α={row.alpha:.1f}\nβ₀={row.beta0:.1f}\nγ={row.gamma:.1f}'
                     for _, row in best_configs.iterrows()]

    bars = ax3.bar(range(len(best_configs)), best_configs['best_fitness'],
                   color=plt.cm.viridis(np.linspace(0, 1, len(best_configs))))
    ax3.set_xlabel('Configuration Rank')
    ax3.set_ylabel('Best Fitness')
    ax3.set_title('Top 10 Parameter Configurations')
    ax3.set_xticks(range(len(best_configs)))
    ax3.set_xticklabels([f'{i + 1}' for i in range(len(best_configs))])

    for i, (_, row) in enumerate(best_configs.iterrows()):
        ax3.text(i, row['best_fitness'] + max(best_configs['best_fitness']) * 0.01,
                 f'α={row.alpha:.1f}\nβ₀={row.beta0:.1f}\nγ={row.gamma:.1f}',
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_fireflies_positions(df, bounds=(-5.12, 5.12)):
    if 'final_positions' not in df.columns:
        print("DataFrame must contain 'final_positions' column")
        return

    gamma_values = sorted(df['gamma'].unique())
    n_gamma = len(gamma_values)

    cols = min(3, n_gamma)
    rows = (n_gamma + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_gamma == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, 12))

    for i, gamma in enumerate(gamma_values):
        ax = axes[i]

        gamma_subset = df[df['gamma'] == gamma]

        all_positions = []

        for idx, row in gamma_subset.iterrows():
            alpha, beta0 = row['alpha'], row['beta0']
            final_positions_runs = row['final_positions']
            best_position = row['best_position']

            config_positions = []
            for run_positions in final_positions_runs:
                config_positions.extend(run_positions)

            if config_positions:
                config_positions = np.array(config_positions)
                all_positions.append({
                    'positions': config_positions,
                    'best_pos': best_position,
                    'label': f'α={alpha:.1f},β₀={beta0:.1f}',
                    'fitness': row['best_fitness']
                })

        if not all_positions:
            ax.text(0.5, 0.5, 'No position data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f'γ = {gamma:.1f}')
            continue

        all_positions.sort(key=lambda x: x['fitness'])

        for j, config_data in enumerate(all_positions):
            positions = config_data['positions']
            best_pos = config_data['best_pos']
            label = config_data['label']

            if positions.shape[1] >= 2:
                ax.scatter(positions[:, 0], positions[:, 1],
                           c=[colors[j % len(colors)]], alpha=0.6, s=20,
                           label=f'{label} (f={config_data["fitness"]:.3f})')

                ax.scatter(best_pos[0], best_pos[1],
                           c='red', marker='*', s=100,
                           edgecolors='black', linewidth=1)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'γ = {gamma:.1f} (Convergence Patterns)')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(bounds[0] - 0.5, bounds[1] + 0.5)
        ax.set_ylim(bounds[0] - 0.5, bounds[1] + 0.5)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax.scatter(0, 0, c='gold', marker='x', s=150,
                   linewidth=3, label='Global Optimum')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
