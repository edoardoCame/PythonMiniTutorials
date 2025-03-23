import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import dask.dataframe as dd
from datetime import datetime

def bollinger_backtest(dask_df, lookback, sdev):
    """
    Backtests a Bollinger Band strategy with limit orders.
    
    Parameters:
    dask_df (dask.dataframe): DataFrame with OHLC data and Timestamp
    lookback (int): Lookback window for Bollinger Bands calculation
    sdev (float): Standard deviation multiplier for bands
    
    Returns:
    dict: Performance statistics
    """
    # Make a copy to avoid modifying the original
    df = dask_df.copy()
    
    # Convert Timestamp to datetime
    df['Timestamp'] = dd.to_datetime(df['Timestamp'])
    
    # Compute the DataFrame to bring it into memory before calculating indicators
    # This fixes the "unknown divisions" error with rolling operations
    df = df.compute()
    
    # Extract hour for trading time filter
    df['Hour'] = df['Timestamp'].dt.hour
    
    # Calculate Bollinger Bands (now using pandas DataFrame)
    df['SMA'] = df['Close'].rolling(lookback).mean()
    df['STD'] = df['Close'].rolling(lookback).std()
    df['Upper'] = df['SMA'] + sdev * df['STD']
    df['Lower'] = df['SMA'] - sdev * df['STD']
    
    # Shift bands to avoid look-ahead bias
    df['Upper_prev'] = df['Upper'].shift(1)
    df['Lower_prev'] = df['Lower'].shift(1)
    df['SMA_prev'] = df['SMA'].shift(1)
    
    df = df.dropna()  # Drop NaN rows after calculating indicators
    
    # Convert to numpy arrays for Numba
    hours = df['Hour'].values
    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    upper_bands = df['Upper_prev'].values
    lower_bands = df['Lower_prev'].values
    middle_bands = df['SMA_prev'].values
    
    # Use Numba for the backtesting logic with custom fee structure
    positions, entry_prices, returns_pips = backtest_strategy_custom_fee(
        hours, high_prices, low_prices, close_prices, 
        upper_bands, lower_bands, middle_bands
    )
    
    # Add results to dataframe
    df['Position'] = positions
    df['Entry_Price'] = entry_prices
    df['Returns_Pips'] = returns_pips
    
    # Convert pips to percentage for equity curve
    pip_value = 0.0001  # Assuming 4 decimal places for FX
    df['Returns_Pct'] = df['Returns_Pips'] * pip_value
    df['Cum_Returns_Pct'] = df['Returns_Pct'].cumsum() * 100  # Convert to percentage
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df['Timestamp'], df['Cum_Returns_Pct'])
    plt.title('Bollinger Band Strategy - Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Returns (%)')
    plt.grid(True)
    plt.show()
    
    # Calculate statistics
    total_return_pct = df['Cum_Returns_Pct'].iloc[-1]
    avg_return_pips = df['Returns_Pips'][df['Returns_Pips'] != 0].mean() if len(df['Returns_Pips'][df['Returns_Pips'] != 0]) > 0 else 0
    
    # Calculate Sharpe ratio
    returns_std = df['Returns_Pct'].std()
    sharpe_ratio = df['Returns_Pct'].mean() / returns_std * np.sqrt(252) if returns_std != 0 else 0
    
    # Fixed drawdown calculation - no longer multiplying by 100 again
    df['Peak'] = df['Cum_Returns_Pct'].cummax()
    df['Drawdown'] = df['Cum_Returns_Pct'] - df['Peak']  # Already in percentage
    max_drawdown = df['Drawdown'].min()
    
    # Count number of trades
    n_trades = len(df[df['Returns_Pips'] != 0])
    
    # Plot average return by hour - improved version
    plt.figure(figsize=(12, 6))
    
    # Filter only trading hours (22-7)
    trading_hours = list(range(22, 24)) + list(range(0, 8))
    hourly_returns = df.groupby('Hour')['Returns_Pips'].mean()
    hourly_returns = hourly_returns.loc[trading_hours]
    
    ax = hourly_returns.plot(kind='bar', color='steelblue', edgecolor='black')
    
    # Enhance the plot appearance
    plt.title('Average Return (pips) by Hour (Trading Hours Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Return (pips)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0, fontsize=10)
    
    # Add values above each bar
    for i, v in enumerate(hourly_returns):
        ax.text(i, v + (0.1 if v >= 0 else -0.3), 
                f'{v:.2f}', 
                ha='center', 
                fontweight='bold')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    stats = {
        'Total Return (%)': total_return_pct,
        'Average Return (pips)': avg_return_pips,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown,
        'Number of Trades': n_trades
    }
    
    print("Strategy Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    return stats

@jit(nopython=True)
def backtest_strategy_custom_fee(hours, high_prices, low_prices, close_prices, 
                      upper_bands, lower_bands, middle_bands):
    """
    Numba-optimized function to run the backtest using limit orders with custom fee structure
    
    Fee structure:
    - From 20 to 23: 3 pips spread
    - All other hours: 1.2 pips spread
    """
    n = len(hours)
    positions = np.zeros(n)
    entry_prices = np.zeros(n)
    returns = np.zeros(n)
    
    in_position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(1, n):
        # Check if we're in the trading hours (22-7)
        if not (hours[i] >= 22 or hours[i] <= 7):
            positions[i] = in_position
            entry_prices[i] = entry_price
            continue
        
        # Check for exit if in position
        if in_position == 1:  # Long position
            if high_prices[i] >= middle_bands[i]:  # Exit long at middle band
                returns[i] = (middle_bands[i] - entry_price) * 10000  # Convert to pips
                in_position = 0
                entry_price = 0.0
        elif in_position == -1:  # Short position
            if low_prices[i] <= middle_bands[i]:  # Exit short at middle band
                returns[i] = (entry_price - middle_bands[i]) * 10000  # Convert to pips
                in_position = 0
                entry_price = 0.0
        
        # Check for entry if not in position
        if in_position == 0:
            # Apply custom fee structure based on hour
            if hours[i] >= 20 and hours[i] <= 23:
                spread_fee = 3.0  # 3 pips from 20-23
            else:
                spread_fee = 1.2  # 1.2 pips for all other hours
                
            # Long entry: price crosses below lower band (limit buy order)
            if low_prices[i] <= lower_bands[i]:
                in_position = 1
                entry_price = lower_bands[i]
                returns[i] -= spread_fee  # Apply appropriate spread commission
            
            # Short entry: price crosses above upper band (limit sell order)
            elif high_prices[i] >= upper_bands[i]:
                in_position = -1
                entry_price = upper_bands[i]
                returns[i] -= spread_fee  # Apply appropriate spread commission
        
        positions[i] = in_position
        entry_prices[i] = entry_price
    
    return positions, entry_prices, returns


# Load data
df = dd.read_csv('EURGBP_polygon.csv')

# Run backtest
stats = bollinger_backtest(dask_df=df, lookback=120, sdev=4)