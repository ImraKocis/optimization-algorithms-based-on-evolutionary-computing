from matplotlib import pyplot as plt
import seaborn as sns


def average_time_performance(results_df):
    time_pivot = results_df.groupby(['c1', 'c2'])['time'].mean().unstack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(time_pivot,
                annot=True,
                fmt='.3f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Average Time (seconds)'})
    plt.title('Average Execution Time Heatmap')
    plt.xlabel('c2 (Social Component)')
    plt.ylabel('c1 (Cognitive Component)')
    plt.tight_layout()
    plt.show()

