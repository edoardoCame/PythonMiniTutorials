import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the grid search results from the CSV file
results_df = pd.read_csv('/home/edoardo/Desktop/python_dir/bb_grid_search_results.csv')

def plot_bb_heatmap(results_df, metric='avg_pips_per_trade', save_path=None):
    """
    Plot a heatmap of Bollinger Bands strategy parameter optimization results.
    
    Args:
        results_df: DataFrame with grid search results
        metric: Which metric to visualize on the heatmap (default: avg_pips_per_trade)
        save_path: If provided, save the figure to this path
    """
    if results_df.empty:
        print("No results to plot")
        return
    
    # Create a pivot table for the heatmap
    pivot = results_df.pivot(index='lookback', columns='sdev', values=metric)
    
    # Set up the figure
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap for the heatmap
    # Red for negative values, green for positive values
    cmap = sns.diverging_palette(h_neg=10, h_pos=120, s=99, l=55, sep=3, as_cmap=True)
    
    # Find the maximum absolute value for symmetric color scaling
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    
    # Create the heatmap with seaborn
    ax = sns.heatmap(
        pivot, 
        annot=True,           # Show values in each cell
        fmt='.1f',            # Format for the annotations
        cmap=cmap,            # Use our custom colormap
        center=0,             # Center the colormap at zero
        vmin=-vmax,           # Set min value for color scale
        vmax=vmax,            # Set max value for color scale
        linewidths=0.5,       # Add grid lines between cells
        cbar_kws={'label': metric}  # Add a label to the color bar
    )
    
    # Find the best parameter combination
    best_row_idx, best_col_idx = np.unravel_index(pivot.values.argmax(), pivot.shape)
    best_lookback = pivot.index[best_row_idx]
    best_sdev = pivot.columns[best_col_idx]
    best_value = pivot.values[best_row_idx, best_col_idx]
    
    # Highlight the best cell
    ax.add_patch(plt.Rectangle((best_col_idx, best_row_idx), 1, 1, fill=False, 
                              edgecolor='black', lw=3, clip_on=False))
    
    # Add a title and labels
    metric_title = "Average Profit per Trade (pips)" if metric == "avg_pips_per_trade" else metric.replace("_", " ").title()
    plt.title(f'Bollinger Bands Strategy Optimization: {metric_title}', fontsize=15)
    plt.xlabel('Standard Deviation', fontsize=12)
    plt.ylabel('Lookback Period', fontsize=12)
    
    # Add text with best parameters
    plt.figtext(0.02, 0.02, f'Best Parameters: Lookback={best_lookback}, SD={best_sdev}, {metric}={best_value:.1f}', 
                fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

# Plot heatmap for average pips per trade
plot_bb_heatmap(results_df, metric='avg_pips_per_trade', 
                save_path='/home/edoardo/Desktop/python_dir/bb_optimization_heatmap.png')

# You can also create heatmaps for other metrics
# Uncomment to see win rate heatmap
# plot_bb_heatmap(results_df, metric='win_rate', 
#                 save_path='/home/edoardo/Desktop/python_dir/bb_winrate_heatmap.png')

# Uncomment to see profit factor heatmap
# plot_bb_heatmap(results_df, metric='profit_factor',
#                save_path='/home/edoardo/Desktop/python_dir/bb_profitfactor_heatmap.png')

print("Done! Check the saved heatmap visualization.")