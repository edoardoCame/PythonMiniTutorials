import numpy as np
import pandas as pd
from numba import jit, njit, prange
import datetime
import matplotlib.pyplot as plt



import dask.dataframe as dd

def bollingerbands_backtest(df, lookback=20, sdev=2.0, sl_percentage=0.01, use_custom_spread=True):
    """
    Backtest a Bollinger Bands trading strategy on OHLC data using Numba for optimization.
    
    Strategy:
    - Long if Close < Lower band, exit at middle band
    - Short if Close > upper band, exit at middle band
    - Only trade between 20:00 and 05:00
    - Stop loss based on High/Low of candle
    
    Parameters:
    -----------
    df : dask.dataframe.DataFrame
        Dataframe containing OHLC data and Timestamp
    lookback : int, default=20
        Lookback period for calculating Bollinger Bands
    sdev : float, default=2.0
        Standard deviation multiplier for Bollinger Bands
    sl_percentage : float, default=0.01
        Stop loss percentage (0.01 = 1%)
    use_custom_spread : bool, default=True
        If True, uses hourly spread structure (3 pips for 21-23h, 1.2 pips otherwise)
    
    Returns:
    --------
    tuple
        (cum_returns, timestamps) - Array of cumulative returns and timestamps
    """
    # Convert Timestamp to datetime if it's not already
    if df['Timestamp'].dtype != 'datetime64[ns]':
        df['Timestamp'] = dd.to_datetime(df['Timestamp'])
    
    # Extract hour from Timestamp for time filtering
    df['hour'] = df['Timestamp'].dt.hour
    
    # Compute to convert from dask to pandas
    df = df.compute()
    
    # Save timestamps for later use in rolling returns
    timestamps = df['Timestamp'].values
    
    # Prepare arrays for numba
    Opens = df['Open'].values.astype(np.float64)
    Highs = df['High'].values.astype(np.float64)
    Lows = df['Low'].values.astype(np.float64)
    closes = df['Close'].values.astype(np.float64)
    hours = df['hour'].values.astype(np.int64)
    
    # Run the backtest using numba
    returns, cum_returns, trade_hours = backtest_strategy(Opens, Highs, Lows, closes, hours, 
                                                        lookback, sdev, sl_percentage, use_custom_spread)
    
    # Calculate and print statistics
    calculate_statistics(returns)
    
    # Plot hourly performance
    plot_hourly_performance(returns, trade_hours)
    
    return cum_returns, timestamps

@njit
def calculate_bollinger_bands(closes, lookback, sdev):
    """
    Calculate Bollinger Bands using Numba optimization.
    
    Parameters:
    -----------
    closes : numpy.ndarray
        Array of closing prices
    lookback : int
        Lookback period for calculating moving average
    sdev : float
        Standard deviation multiplier
    
    Returns:
    --------
    tuple
        (upper_band, middle_band, Lower_band)
    """
    n = len(closes)
    upper_band = np.zeros(n)
    middle_band = np.zeros(n)
    Lower_band = np.zeros(n)
    
    for i in range(lookback, n):
        window = closes[i-lookback:i]
        middle_band[i] = np.mean(window)
        std = np.std(window)
        upper_band[i] = middle_band[i] + sdev * std
        Lower_band[i] = middle_band[i] - sdev * std
    
    return upper_band, middle_band, Lower_band

@njit
def backtest_strategy(Opens, Highs, Lows, closes, hours, lookback, sdev, sl_percentage, use_custom_spread):
    """
    Run the backtest using Numba optimization.
    
    Parameters:
    -----------
    Opens, Highs, Lows, closes : numpy.ndarray
        Arrays of OHLC prices
    hours : numpy.ndarray
        Array of hours from Timestamp
    lookback : int
        Lookback period for Bollinger Bands
    sdev : float
        Standard deviation multiplier
    sl_percentage : float
        Stop loss percentage
    use_custom_spread : bool
        If True, uses hourly spread structure (3 pips for 21-23h, 1.2 pips otherwise)
    
    Returns:
    --------
    tuple
        (returns, cum_returns, trade_hours)
    """
    n = len(closes)
    upper_band, middle_band, Lower_band = calculate_bollinger_bands(closes, lookback, sdev)
    
    # Initialize arrays for tracking trades and performance
    position = np.zeros(n)  # 1 for long, -1 for short, 0 for no position
    returns = np.zeros(n)
    entry_price = np.zeros(n)
    stop_loss_level = np.zeros(n)
    trade_hours = np.zeros(n, dtype=np.int64)  # Track the hour when each trade is closed
    pip_multiplier = 10000  # Standard for 4-decimal forex pairs
    
    for i in range(lookback + 1, n):
        # Check if we're within trading hours (20:00 to 05:00)
        valid_hour = (hours[i] >= 20) or (hours[i] <= 5)
        
        # Use custom spread structure
        if use_custom_spread:
            # Apply 3 pips spread for hours 21-23, otherwise 1.2 pips
            if hours[i] >= 21 and hours[i] <= 23:
                spread_cost = 3.0 / pip_multiplier
            else:
                spread_cost = 1.2 / pip_multiplier
        else:
            # Default spread (kept for backward compatibility)
            spread_cost = 3.0 / pip_multiplier
        
        # Continue with position from previous bar
        position[i] = position[i-1]
        entry_price[i] = entry_price[i-1]
        stop_loss_level[i] = stop_loss_level[i-1]
        
        # Check if we're in a position
        if position[i] == 1:  # Long position
            # Check for stop loss hit
            if Lows[i] <= stop_loss_level[i]:
                # Stop loss triggered, Close position
                returns[i] = (stop_loss_level[i] - entry_price[i])  # Return in price difference
                trade_hours[i] = hours[i]  # Record the hour of this trade
                position[i] = 0
            # Check for target (middle band) hit
            elif closes[i] >= middle_band[i]:
                # Take profit at middle band
                returns[i] = (closes[i] - entry_price[i])  # Return in price difference
                trade_hours[i] = hours[i]  # Record the hour of this trade
                position[i] = 0
        
        elif position[i] == -1:  # Short position
            # Check for stop loss hit
            if Highs[i] >= stop_loss_level[i]:
                # Stop loss triggered, Close position
                returns[i] = (entry_price[i] - stop_loss_level[i])  # Return in price difference
                trade_hours[i] = hours[i]  # Record the hour of this trade
                position[i] = 0
            # Check for target (middle band) hit
            elif closes[i] <= middle_band[i]:
                # Take profit at middle band
                returns[i] = (entry_price[i] - closes[i])  # Return in price difference
                trade_hours[i] = hours[i]  # Record the hour of this trade
                position[i] = 0
        
        # If we're not in a position and within trading hours, check for new entry signals
        elif position[i] == 0 and valid_hour:
            if closes[i] < Lower_band[i]:
                # Go long
                position[i] = 1
                entry_price[i] = closes[i]
                stop_loss_level[i] = entry_price[i] * (1 - sl_percentage)  # Set stop loss percentage beLow entry
                # Apply spread cost at entry for long position
                returns[i] = -spread_cost  # Deduct spread cost immediately
                
            elif closes[i] > upper_band[i]:
                # Go short
                position[i] = -1
                entry_price[i] = closes[i]
                stop_loss_level[i] = entry_price[i] * (1 + sl_percentage)  # Set stop loss percentage above entry
                # Apply spread cost at entry for short position
                returns[i] = -spread_cost  # Deduct spread cost immediately
    
    # Calculate cumulative returns
    cum_returns = np.zeros(n)
    cum_returns[0] = 1.0  # Start with $1
    
    for i in range(1, n):
        if returns[i] != 0:  # Only update on trade closures or entries with spread cost
            cum_returns[i] = cum_returns[i-1] * (1 + returns[i])
        else:
            cum_returns[i] = cum_returns[i-1]
    
    return returns, cum_returns, trade_hours

def calculate_statistics(returns, pip_multiplier=10000):
    """
    Calculate and print trade statistics in pips.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of returns in price difference
    pip_multiplier : int, default=10000
        Multiplier to convert price difference to pips (10000 for 4-decimal forex pairs)
    """
    # Convert price returns to pip values
    pip_returns = returns * pip_multiplier
    
    # Filter out zero returns (no trades)
    non_zero_returns = pip_returns[pip_returns != 0]
    
    if len(non_zero_returns) == 0:
        print("No trades were executed.")
        return
    
    # Calculate statistics
    total_pips = np.sum(non_zero_returns)
    win_rate = np.sum(non_zero_returns > 0) / len(non_zero_returns)
    avg_win_pips = np.mean(non_zero_returns[non_zero_returns > 0]) if np.any(non_zero_returns > 0) else 0
    avg_loss_pips = np.mean(non_zero_returns[non_zero_returns < 0]) if np.any(non_zero_returns < 0) else 0
    profit_factor = abs(np.sum(non_zero_returns[non_zero_returns > 0]) / 
                      np.sum(non_zero_returns[non_zero_returns < 0])) if np.any(non_zero_returns < 0) else float('inf')
    
    # Calculate risk-adjusted returns
    annual_factor = 252  # Trading days in a year
    sharpe_ratio = np.mean(non_zero_returns) / np.std(non_zero_returns) * np.sqrt(annual_factor) if np.std(non_zero_returns) > 0 else 0
    
    # Fixed drawdown calculation
    # Start with initial capital of 1.0
    equity = 1.0
    equity_curve = [equity]
    
    for ret in returns:
        if ret != 0:  # Only update on trade events
            equity *= (1 + ret)  # Apply return directly without pip conversion
        equity_curve.append(equity)
    
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(np.min(drawdown)) * 100
    
    # Calculate Sortino ratio
    negative_returns = non_zero_returns[non_zero_returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1
    sortino_ratio = np.mean(non_zero_returns) / downside_std * np.sqrt(annual_factor) if downside_std > 0 else 0
    
    # Print statistics
    print(f"\n=== Bollinger Bands Strategy Performance ===")
    print(f"Total Trades: {len(non_zero_returns)}")
    print(f"Net Pips: {total_pips:.2f}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win (pips): {avg_win_pips:.2f}")
    print(f"Average Loss (pips): {avg_loss_pips:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print("==========================================")

def plot_hourly_performance(returns, trade_hours, pip_multiplier=10000):
    """
    Plot the average profit distribution by hour for trades executed between 20:00 and 05:00.
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Array of returns in price difference
    trade_hours : numpy.ndarray
        Array of hours when trades were executed
    pip_multiplier : int, default=10000
        Multiplier to convert price difference to pips
    """
    # Convert returns to pips for easier interpretation
    # In forex, small price movements are measured in pips (percentage in point)
    # For 4-decimal pairs like EUR/GBP, 1 pip = 0.0001, so we multiply by 10000
    pip_returns = returns * pip_multiplier
    
    # Create hour labels from 20 to 5 - these are the trading hours we're analyzing
    # The strategy only trades during these night/early morning hours, so we focus on them
    hour_labels = list(range(20, 24)) + list(range(0, 6))
    hour_profits = {hour: [] for hour in hour_labels}
    
    # Group returns by hour to see performance differences across different times
    # This helps identify if certain hours consistently outperform others
    for i in range(len(returns)):
        if returns[i] != 0:  # Only consider actual trades (non-zero returns)
            hour = trade_hours[i]
            if hour in hour_labels:  # Only include trades within our trading hours window
                hour_profits[hour].append(pip_returns[i])
    
    # Calculate average profit for each hour to identify the most profitable times to trade
    # Using mean gives us a central tendency measure for each hour's performance
    avg_profits = []
    for hour in hour_labels:
        if len(hour_profits[hour]) > 0:
            avg_profits.append(np.mean(hour_profits[hour]))
        else:
            avg_profits.append(0)  # No trades for this hour, set to zero
    
    # Create the bar plot visualization with a significant size for readability
    # Horizontal size (12) accommodates all hour labels, vertical size (6) shows differences clearly
    plt.figure(figsize=(12, 6))
    bars = plt.bar([str(h) for h in hour_labels], avg_profits)
    
    # Color bars based on profit/loss for immediate visual distinction
    # Green bars represent profitable hours, red bars represent losing hours
    # This visual cue helps quickly identify patterns in hourly performance
    for i, bar in enumerate(bars):
        if avg_profits[i] > 0:
            bar.set_color('green')  # Profitable hours
        else:
            bar.set_color('red')    # Unprofitable hours
    
    # Add informative title and labels to clarify what the chart represents
    plt.title('Average Profit Distribution by Hour (in pips)')
    plt.xlabel('Hour')
    plt.ylabel('Average Profit (pips)')
    
    # Add grid lines on Y-axis for easier quantitative comparison between hours
    # Using dashed lines with reduced opacity to avoid visual clutter
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add data labels on top of each bar for precise values
    # Position positive values above the bar and negative values below for clarity
    # This eliminates the need to estimate values from the axis
    for i, v in enumerate(avg_profits):
        plt.text(i, v + (0.1 if v >= 0 else -3), 
                 f"{v:.2f}", 
                 ha='center',  # Horizontally centered on the bar
                 va='bottom' if v >= 0 else 'top')  # Positioning depends on value sign
    
    # Adjust layout to ensure all elements are visible and properly spaced
    plt.tight_layout()
    plt.show()

def plot_rolling_returns(cum_returns, timestamps, window=60, figsize=(12, 6)):
    """
    Plot rolling returns for a specified window period.
    
    Parameters:
    -----------
    cum_returns : numpy.ndarray
        Array of cumulative returns
    timestamps : numpy.ndarray
        Array of timestamps corresponding to returns
    window : int, default=20
        Rolling window size (20 days ~= 1 month of trading days)
    figsize : tuple, default=(12, 6)
        Figure size for the plot
    """
    # Ensure data alignment between returns and timestamps
    # This is crucial for accurate time-based analysis and prevents index mismatches
    if len(cum_returns) != len(timestamps):
        # If lengths don't match, truncate both arrays to the shorter length
        # This preserves the temporal relationship between returns and timestamps
        min_len = min(len(cum_returns), len(timestamps))
        cum_returns = cum_returns[:min_len]
        timestamps = timestamps[:min_len]
    
    # Convert timestamps to pandas datetime and create a time-indexed Series
    # This transformation enables advanced time series operations and proper date formatting
    time_index = pd.to_datetime(timestamps)
    returns_series = pd.Series(cum_returns, index=time_index)
    
    # Calculate log returns (period-to-period percentage changes)
    # Log returns are preferred in financial analysis because:
    #  1. They're additive over time (unlike simple returns)
    #  2. They're more normally distributed
    #  3. They better represent compounding effects
    log_returns = returns_series.pct_change().fillna(0)
    
    # Calculate rolling returns using a sliding window approach
    # This shows the strategy's performance over consistent time periods, revealing:
    #  - Performance consistency
    #  - Volatility trends
    #  - Cyclical patterns
    rolling_returns = (1 + log_returns).rolling(window=window).apply(
        lambda x: np.prod(x) - 1, raw=True  # Calculate compound growth over the window
    ) * 100  # Convert to percentage for more intuitive interpretation
    
    # Create descriptive window label based on common financial reporting periods
    # This adds context to the chart by relating the window to standard time frames
    if window == 20:
        window_label = f"{window}-Day (Monthly)"  # ~1 month of trading days
    elif window == 60:
        window_label = f"{window}-Day (Quarterly)"  # ~3 months of trading days
    elif window == 252:
        window_label = f"{window}-Day (Annual)"  # ~1 year of trading days
    else:
        window_label = f"{window}-Day"  # Custom period
    
    # Initialize the plot with appropriate dimensions for detailed analysis
    plt.figure(figsize=figsize)
    
    # Plot the rolling returns as a continuous line
    # Blue is chosen for visibility and as a neutral color (not suggesting profit/loss)
    plt.plot(rolling_returns, color='blue')
    
    # Add a horizontal line at y=0 to clearly show positive vs negative performance periods
    # This reference line is crucial for quickly identifying winning and losing periods
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add descriptive title and labels to make the chart self-explanatory
    plt.title(f'{window_label} Rolling Returns')
    plt.xlabel('Date')  # X-axis represents time
    plt.ylabel('Rolling Returns (%)')  # Y-axis shows percentage returns
    
    # Add grid for easier value reading and comparison across time periods
    # Low alpha keeps the grid unobtrusive while still providing reference lines
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show dates properly angled for readability
    # This prevents date labels from overlapping, especially in dense time series
    plt.gcf().autofmt_xdate()
    
    # Add key statistical measures at the bottom of the chart for quick performance assessment
    # These summary statistics help evaluate the strategy's overall characteristics:
    mean_return = rolling_returns.mean()
    plt.figtext(0.15, 0.02, f'Mean: {mean_return:.2f}%', ha='left')  # Average performance
    plt.figtext(0.35, 0.02, f'Min: {rolling_returns.min():.2f}%', ha='left')  # Worst period
    plt.figtext(0.55, 0.02, f'Max: {rolling_returns.max():.2f}%', ha='left')  # Best period
    plt.figtext(0.75, 0.02, f'Std Dev: {rolling_returns.std():.2f}%', ha='left')  # Volatility measure
    
    # Adjust layout to ensure all elements are visible and well-spaced
    plt.tight_layout()
    
    # Add extra bottom margin to accommodate the summary statistics
    plt.subplots_adjust(bottom=0.15)
    
    # Display the plot
    plt.show()

# Load sample data
import dask.dataframe as dd
df = dd.read_csv('EURGBP_polygon.csv')

# Run backtest with custom spread structure
cum_returns, timestamps = bollingerbands_backtest(df, lookback=120, sdev=4.0, sl_percentage=0.1, use_custom_spread=True)

# Plot the results with log scale
plt.figure(figsize=(10, 6))  # Set figure size for optimal viewing
plt.plot(cum_returns)  # Plot the equity curve showing strategy performance over time

# Add descriptive title that includes key strategy information
plt.title('Bollinger Bands Strategy Performance (Custom Spread Structure)')

# Label axes for clarity
plt.xlabel('Number of Bars')  # X-axis represents time progression in terms of price bars
plt.ylabel('Cumulative Returns')  # Y-axis shows the growth of $1 invested

# Use logarithmic scale for the y-axis
# Log scale is crucial for equity curves because:
#  1. It shows percentage changes equally (10% up and 10% down have equal visual weight)
#  2. It prevents recent performance from dominating the chart visually
#  3. It makes the growth rate (slope) directly comparable across different time periods
plt.yscale('log')

# Add grid lines for both major and minor ticks to make the log scale more readable
# Alpha is reduced to avoid visual clutter while still providing reference lines
plt.grid(True, which="both", ls="-", alpha=0.3)

# Display the plot
plt.show()

# Plot rolling returns with a 60-day (quarterly) window
# This helps assess the strategy's consistency and identify performance trends
plot_rolling_returns(cum_returns, timestamps, window=60)

# Uncomment these to try different window periods:
# Plot quarterly rolling returns (using 60 trading days)
# plot_rolling_returns(cum_returns, timestamps, window=60)

# Plot annual rolling returns (using 252 trading days)
# plot_rolling_returns(cum_returns, timestamps, window=252)