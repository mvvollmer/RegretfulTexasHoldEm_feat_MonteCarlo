import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_cumulative_winnings(total_winnings, players, n_iterations):
    """
    Plot cumulative winnings for multiple players across iterations with confidence intervals.
    
    Parameters:
    total_winnings: list of numpy arrays, each array containing trial results
    players: list of player names
    n_iterations: int, number of iterations/rounds
    """
    # Calculate cumulative winnings for each trial
    total_cumulative_winnings = []
    for trial in total_winnings:
        # Ensure each trial has exactly n_iterations rows
        if len(trial) > n_iterations:
            trial = trial[:n_iterations]
        elif len(trial) < n_iterations:
            # Pad with zeros if necessary
            padding = np.zeros((n_iterations - len(trial), trial.shape[1]))
            trial = np.vstack([trial, padding])
            
        cumulative_winnings = np.cumsum(trial, axis=0)
        total_cumulative_winnings.append(cumulative_winnings)

    # Convert to NumPy array for easier manipulation
    total_cumulative_winnings = np.array(total_cumulative_winnings)

    # Compute mean and standard error across trials
    mean_cumulative_winnings = np.mean(total_cumulative_winnings, axis=0)
    stderr_cumulative_winnings = stats.sem(total_cumulative_winnings, axis=0)  # Using stats.sem instead

    # Create the plot
    plt.figure(figsize=(12, 8))
    x = np.arange(mean_cumulative_winnings.shape[0])  # Use actual length of data

    # Color map for different players
    colors = plt.cm.tab10(np.linspace(0, 1, len(players)))

    for i, (player, color) in enumerate(zip(players, colors)):
        mean = mean_cumulative_winnings[:, i]
        stderr = stderr_cumulative_winnings[:, i]
        lower_bound = mean - 1.96 * stderr
        upper_bound = mean + 1.96 * stderr

        # Ensure all arrays have the same length as x
        mean = mean[:len(x)]
        lower_bound = lower_bound[:len(x)]
        upper_bound = upper_bound[:len(x)]

        # Plot mean line with custom color
        plt.plot(x, mean, label=player, color=color, linewidth=2)
        
        # Fill between the confidence interval
        plt.fill_between(x, lower_bound, upper_bound, color=color, alpha=0.2)

    # Customize the plot
    plt.xlabel('Rounds', fontsize=12)
    plt.ylabel('Cumulative Winnings', fontsize=12)
    plt.title('Average Cumulative Player Winnings with 95% Confidence Interval', 
              fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    return plt.gcf()