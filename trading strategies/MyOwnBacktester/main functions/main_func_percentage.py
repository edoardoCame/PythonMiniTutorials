#Create a backtesting function that backtests based on Bollinger Bands
from numba import njit #We import the njit decorator from the numba library
@njit(fastmath=True) #We use the njit decorator to compile the function to machine code, which makes it faster
def backtest_bollinger_bands(data, window, num_std_devs, fees_percentage=0.015, starting_cash=10000):
    
    
    #-------------------Bollinger Bands-------------------

    #We calculate the rolling mean:
    middle_band = np.zeros(len(data)) #We create an array of zeros to store the rolling mean
    for i in range(window, len(data)):
        middle_band[i] = np.mean(data[i-window:i]) #We calculate the rolling mean for each day

    #We calculate the rolling standard deviation:
    rolling_std = np.zeros(len(data))
    for i in range(window, len(data)):
        rolling_std[i] = np.std(data[i-window:i]) #We calculate the rolling standard deviation for each day

    #We calculate the upper and lower bands:
    upper_band = middle_band + num_std_devs * rolling_std
    lower_band = middle_band - num_std_devs * rolling_std


    

    #Strategy's logic:
    positions = np.zeros(len(data)) #We create an array of zeros to store the positions
    
    for i in range(window, len(data)): #from the second row to the last row of the data
        if data[i] < lower_band[i] and positions[i-1] == 0:
            positions[i] = 1 #long if the price is below the lower band and we don't have a position
        elif data[i] > upper_band[i] and positions[i-1] == 0:
            positions[i] = -1 #short if the price is above the upper band and we don't have a position
        elif data[i] > middle_band[i] and positions[i-1] == 1:
            positions[i] = 0 #close long position if the price is above the middle band and we have a long position
        elif data[i] < middle_band[i] and positions[i-1] == -1:
            positions[i] = 0 #close short position if the price is below the middle band and we have a short position
        else:
            positions[i] = positions[i-1] #carry on with the previous position if none of the conditions are met;

    #We calculate the returns of the strategy:
    
    market_returns = np.zeros(len(data)) #We create an array of zeros to store the market returns
    for i in range(1, len(data)):
        market_returns[i] = (data[i] - data[i-1]) / data[i-1] #We calculate the market returns for each day
    
    strategy_returns = np.zeros(len(data)) #We create an array of zeros to store the strategy returns
    for i in range(1, len(data)):
        strategy_returns[i] = positions[i-1] * market_returns[i] #We calculate the strategy returns for each day, shifting the positions array by 1 day
    

    #Calculate equity with for loop
    equity = np.zeros(len(data))
    equity[0] = starting_cash

    for i in range(1, len(data)):
        if positions[i-1] == positions[i]:
            equity[i] = equity[i-1] * (1 + strategy_returns[i])
        else:
            equity[i] = equity[i-1] * (1 + strategy_returns[i]) - equity[i-1] * fees_percentage / 100


    return equity