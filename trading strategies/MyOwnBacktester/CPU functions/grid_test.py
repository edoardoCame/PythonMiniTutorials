import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dask.dataframe as dd
from tqdm import tqdm
import os
import sys

# Import the backtest_bb_strategy from the bb_limit_orders module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bb_limit_orders import backtest_bb_strategy

def grid_search_bb_parameters(data_path, lookback_range, sdev_range):
    """
    Perform a grid search to find the optimal Bollinger Bands parameters.
    
    Args:
        data_path: Path to the CSV file with OHLC data
        lookback_range: List of lookback periods to test
        sdev_range: List of standard deviation values to test
    
    Returns:
        DataFrame with results of all parameter combinations
        DataFrame with best parameter combination
    """
    print(f"Loading data from {data_path}...")
    data = dd.read_csv(data_path, parse_dates=['Timestamp'])
    data = data.set_index('Timestamp')
    
    # Create a grid of all parameter combinations
    results = []
    
    print(f"Running grid search with {len(lookback_range) * len(sdev_range)} parameter combinations...")
    
    # Use tqdm for a progress bar
    total_combinations = len(lookback_range) * len(sdev_range)
    progress_counter = 0
    
    for lookback in lookback_range:
        for sdev in sdev_range:
            progress_counter += 1
            print(f"Testing combination {progress_counter}/{total_combinations}: lookback={lookback}, sdev={sdev}")
            
            try:
                # Run the backtest with current parameters (with plots disabled)
                _, trades_df = backtest_bb_strategy(data, lookback, sdev, plot=False)
                
                # Calculate performance metrics
                total_pips = np.sum(trades_df['pip_return'])
                num_trades = len(trades_df)
                win_rate = np.sum(trades_df['pip_return'] > 0) / num_trades if num_trades > 0 else 0
                avg_win = np.mean(trades_df['pip_return'][trades_df['pip_return'] > 0]) if np.any(trades_df['pip_return'] > 0) else 0
                avg_loss = np.mean(trades_df['pip_return'][trades_df['pip_return'] < 0]) if np.any(trades_df['pip_return'] < 0) else 0
                avg_pips_per_trade = total_pips / num_trades if num_trades > 0 else 0
                
                # Store results
                results.append({
                    'lookback': lookback,
                    'sdev': sdev,
                    'total_pips': total_pips,
                    'avg_pips_per_trade': avg_pips_per_trade,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0
                })
                
            except Exception as e:
                print(f"Error with parameters lookback={lookback}, sdev={sdev}: {e}")
                continue
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find the best parameter combination based on total pips
    if not results_df.empty:
        best_params = results_df.loc[results_df['total_pips'].idxmax()]
    else:
        best_params = pd.Series({'lookback': None, 'sdev': None, 'total_pips': 0})
    
    return results_df, best_params

def plot_heatmap(results_df, metric='total_pips'):
    """
    Plot a heatmap of the grid search results.
    
    Args:
        results_df: DataFrame with grid search results
        metric: Metric to plot (default: total_pips)
    """
    if results_df.empty:
        print("No results to plot")
        return
    
    # Create a pivot table for the heatmap
    pivot = results_df.pivot(index='lookback', columns='sdev', values=metric)
    
    plt.figure(figsize=(14, 10))
    
    # Create heatmap with seaborn
    sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.1f')
    
    plt.title(f'Grid Search Results: {metric}')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Lookback Period')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define the parameter ranges to test
    lookback_range = range(500, 1440, 30)  # 8, 10, 12, ..., 24
    # Use a list for sdev_range since range() doesn't support float steps
    sdev_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
    
    # Path to the data file
    data_path = '/home/edoardo/Desktop/python_dir/data/EURGBP_polygon.csv'
    
    # Run the grid search
    results_df, best_params = grid_search_bb_parameters(data_path, lookback_range, sdev_range)
    
    # Print the best parameters
    print("\nBest Parameter Combination:")
    print(f"Lookback: {best_params['lookback']}")
    print(f"Standard Deviation: {best_params['sdev']}")
    print(f"Total Pips: {best_params['total_pips']:.2f}")
    print(f"Number of Trades: {best_params['num_trades']}")
    print(f"Win Rate: {best_params['win_rate']:.2%}")
    
    # Plot the heatmap
    plot_heatmap(results_df, metric='total_pips')
    
    # Save results to CSV
    results_df.to_csv("bb_grid_search_results.csv", index=False)
    print("Results saved to bb_grid_search_results.csv")