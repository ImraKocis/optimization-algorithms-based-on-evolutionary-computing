import pandas as pd
from matplotlib import pyplot as plt


def plot_hill_climber_data(data):

    df = pd.DataFrame(data)

    plt.figure(figsize=(15, 10))

    # Plot 1: Raw timing data
    plt.subplot(2, 2, 1)
    plt.plot(df['input_size_max_iter'], df['time_total'], 'o-', label='Actual time')
    plt.xlabel('Problem Size (max iterations)')
    plt.ylabel('Time (seconds)')
    plt.title('Time vs Problem Size')
    plt.grid(True)
    plt.legend()

    # Plot 2: Growth rate
    growth_rates = df['time_total'].pct_change()
    theoretical_growth = df['input_size_max_iter'].mul(df['input_size_local_search']).pct_change()

    plt.subplot(2, 2, 2)
    plt.plot(df['input_size_max_iter'][1:], growth_rates[1:], 'o-', label='Actual growth rate')
    plt.plot(df['input_size_max_iter'][1:], theoretical_growth[1:], '--', label='Theoretical O(m*n)')
    plt.xlabel('Problem Size (max iterations)')
    plt.ylabel('Growth Rate')
    plt.title('Growth Rate Analysis')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
