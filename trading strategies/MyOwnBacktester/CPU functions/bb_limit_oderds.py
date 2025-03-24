import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb
from numba import jit
import dask.dataframe as dd

@jit(nopython=True)
def calculate_bollinger_bands(close_prices, lookback, sdev):
    """
    Calculate Bollinger Bands using Numba for performance
    
    Args:
        close_prices: Array of closing prices
        lookback: Window size for moving average
        sdev: Number of standard deviations
    
    Returns:
        middle_band, upper_band, lower_band arrays
    """
    n = len(close_prices)
    middle_band = np.empty(n)
    upper_band = np.empty(n)
    lower_band = np.empty(n)
    
    # Initialize with NaNs (converted to 0s for Numba compatibility)
    middle_band[:lookback-1] = 0
    upper_band[:lookback-1] = 0
    lower_band[:lookback-1] = 0
    
    # Calculate bands using rolling window
    for i in range(lookback-1, n):
        window = close_prices[i-lookback+1:i+1]
        middle_band[i] = np.mean(window)
        std_dev = np.std(window)
        upper_band[i] = middle_band[i] + sdev * std_dev
        lower_band[i] = middle_band[i] - sdev * std_dev
    
    return middle_band, upper_band, lower_band

@jit(nopython=True)
def backtest_bollinger_bands_strategy(open_prices, high_prices, low_prices, close_prices, 
                                      middle_band, upper_band, lower_band, 
                                      hour_of_day, pip_value=0.0001):
    """
    Backtest the Bollinger Bands strategy using limit orders.
    
    Args:
        open_prices: Array of opening prices
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of closing prices
        middle_band: Array of middle Bollinger Band values
        upper_band: Array of upper Bollinger Band values
        lower_band: Array of lower Bollinger Band values
        hour_of_day: Array containing the hour of each candle
        pip_value: Value of one pip (default 0.0001 for 4-digit pairs)
    
    Returns:
        pip_returns: Array of returns in pips for each trade
        trade_hours: Array of hours when trades were executed
        equity_curve: Cumulative sum of pip_returns
    """
    n = len(close_prices)
    pip_returns = []
    trade_hours = []
    equity_curve = np.zeros(n)
    
    # Trading state variables
    in_long = False
    in_short = False
    entry_price = 0.0
    
    for i in range(1, n):
        # Skip if we don't have a valid Bollinger Band value yet
        if middle_band[i-1] == 0:
            continue
            
        # IMPORTANT: We use i-1 values for the bands to avoid lookahead bias
        # This simulates placing limit orders based on the previous candle's bands
        
        # Check if we're in a position
        if in_long:
            # Check if our exit limit order at middle band got filled
            if high_prices[i] >= middle_band[i-1]:  # Using previous bar's middle band for the limit order
                exit_price = middle_band[i-1]
                pips_gained = (exit_price - entry_price) / pip_value
                pip_returns.append(pips_gained)
                trade_hours.append(hour_of_day[i])
                in_long = False
                equity_curve[i] = equity_curve[i-1] + pips_gained
            else:
                equity_curve[i] = equity_curve[i-1]
                
        elif in_short:
            # Check if our exit limit order at middle band got filled
            if low_prices[i] <= middle_band[i-1]:  # Using previous bar's middle band for the limit order
                exit_price = middle_band[i-1]
                pips_gained = (entry_price - exit_price) / pip_value
                pip_returns.append(pips_gained)
                trade_hours.append(hour_of_day[i])
                in_short = False
                equity_curve[i] = equity_curve[i-1] + pips_gained
            else:
                equity_curve[i] = equity_curve[i-1]
                
        else:
            # Not in a position, check for entry signals
            equity_curve[i] = equity_curve[i-1]
            
            # Long entry: Check if price touched lower band with a limit order
            if low_prices[i] <= lower_band[i-1]:  # Using previous bar's lower band for the limit order
                entry_price = lower_band[i-1]
                in_long = True
                
            # Short entry: Check if price touched upper band with a limit order
            elif high_prices[i] >= upper_band[i-1]:  # Using previous bar's upper band for the limit order
                entry_price = upper_band[i-1]
                in_short = True
    
    return np.array(pip_returns), np.array(trade_hours), equity_curve

def backtest_bb_strategy(data, lookback, sdev):
    """
    Main backtesting function for Bollinger Bands strategy.
    
    Args:
        data: Dask DataFrame with OHLC data
        lookback: Window parameter for the SMA
        sdev: Standard deviation parameter for the Bollinger Bands
    
    Returns:
        DataFrame with backtest results and trades DataFrame
    """
    # Convert Dask DataFrame to Pandas
    df = data.compute()
    
    # Ensure we have datetime index and hour information
    if not isinstance(df.index, pd.DatetimeIndex):
        df['datetime'] = pd.to_datetime(df.index)
        df.set_index('datetime', inplace=True)
    
    # Extract hour of day for analysis
    df['hour'] = df.index.hour
    
    # Prepare arrays for Numba (which doesn't support pandas)
    # Using capitalized column names to match the data file
    open_array = df['Open'].values
    high_array = df['High'].values
    low_array = df['Low'].values
    close_array = df['Close'].values
    hour_array = df['hour'].values
    
    # Calculate Bollinger Bands
    middle_band, upper_band, lower_band = calculate_bollinger_bands(close_array, lookback, sdev)
    
    # Add bands to DataFrame for inspection
    df['middle_band'] = middle_band
    df['upper_band'] = upper_band
    df['lower_band'] = lower_band
    
    # Run the backtest
    pip_returns, trade_hours, equity_curve = backtest_bollinger_bands_strategy(
        open_array, high_array, low_array, close_array, 
        middle_band, upper_band, lower_band, hour_array
    )
    
    # Add equity curve to DataFrame
    df['equity_curve'] = equity_curve
    
    # Calculate performance metrics
    total_pips = np.sum(pip_returns)
    num_trades = len(pip_returns)
    win_rate = np.sum(pip_returns > 0) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(pip_returns[pip_returns > 0]) if np.any(pip_returns > 0) else 0
    avg_loss = np.mean(pip_returns[pip_returns < 0]) if np.any(pip_returns < 0) else 0
    
    # Create a DataFrame for trades
    trades_df = pd.DataFrame({
        'pip_return': pip_returns,
        'hour': trade_hours
    })
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Average profit by hour
    hourly_performance = trades_df.groupby('hour')['pip_return'].mean()
    hourly_performance.plot(kind='bar', ax=axes[0], color='darkblue', alpha=0.7)
    axes[0].set_title('Average Profit in Pips by Hour')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Average Pips')
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Equity curve
    df['equity_curve'].plot(ax=axes[1], color='green')
    axes[1].set_title('Equity Curve (Cumulative Pips)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Pips')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print performance summary
    print(f"Backtest Results for Bollinger Bands Strategy (Lookback: {lookback}, StdDev: {sdev})")
    print(f"Total Pips: {total_pips:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: {avg_win:.2f} pips")
    print(f"Average Loss: {avg_loss:.2f} pips")
    
    return df, trades_df



#######################
# Example usage:
data = dd.read_csv('/home/edoardo/Desktop/python_dir/data/EURGBP_polygon.csv', 
                  parse_dates=['Timestamp'])  # Capital T in Timestamp
data = data.set_index('Timestamp')  # Capital T in Timestamp
results, trades = backtest_bb_strategy(data, lookback=530, sdev=5)
#higher lookbacks lead to higher performance