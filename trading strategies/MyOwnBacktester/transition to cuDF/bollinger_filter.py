

##############################################################################################################
#Here I define all the backtesting functions:
##############################################################################################################

import cudf as cf

def backtest_bollinger_strategy(data, initial_capital=100, fee_percentage=0.0001, 
                                lookback=60, sdev=6, return_series=False, leverage=10,
                                filter=0):
    
    lookback = int(lookback)
    sdev = int(sdev)
    #Here I calculate the bollinger bands first:
    rolling_mean = data['close'].rolling(window=lookback).mean()
    rolling_std = data['close'].rolling(window=lookback).std()
    data['Middle Band'] = rolling_mean
    data['Upper Band'] = rolling_mean + (rolling_std * sdev)
    data['Lower Band'] = rolling_mean - (rolling_std * sdev)

    # Generate initial signals
    data['Signal'] = 0
    data.loc[data['close'] < data['Lower Band'], 'Signal'] = 1
    data.loc[data['close'] > data['Upper Band'], 'Signal'] = -1
    
    # Calculate position changes based on middle band crosses
    data['Above_Middle'] = data['close'] > data['Middle Band']

    data['Middle_Cross_Up'] = (data['Above_Middle'] != data['Above_Middle'].shift(1)) & data['Above_Middle']
    
    data['Middle_Cross_Down'] = (data['Above_Middle'] != data['Above_Middle'].shift(1)) & (~data['Above_Middle']) #~ is NOT operator
    
    # Initialize positions
    data['Position'] = data['Signal']
    
    # Update positions based on middle band crosses
    long_exits = data['Middle_Cross_Down'] & (data['Position'].shift(1) == 1)
    short_exits = data['Middle_Cross_Up'] & (data['Position'].shift(1) == -1)
    
    # Apply position updates
    data.loc[long_exits | short_exits, 'Position'] = 0 #when long/short exits trigger, position is reset to zero;
    
    # Forward fill positions
    data['Position'] = data['Position'].fillna(method='ffill')
    data['Position'] = data['Position'].fillna(0)
    
    # Calculate returns
    data['Returns'] = data['close'].pct_change()
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
    

    # Identify when an order is triggered (when the position changes)
    data['Order Triggered'] = data['Position'] != data['Position'].shift(1)

    # Subtract the fee from the strategy returns on those days
    data.loc[data['Order Triggered'], 'Strategy Returns'] -= (fee_percentage / 100)



    #Filter 1 means higher than 20 or lower than 5, filter 2 means between 5 and 20, filter 0 means no filter;
    if filter == 1:
        #Filter by hours
        data.index = cf.to_datetime(data['timestamp'])   
        data.loc[(data.index.hour > 20) | (data.index.hour < 5), 'Strategy Returns'] = 0
    elif filter == 2:
        data.index = cf.to_datetime(data['timestamp'])   
        data.loc[(data.index.hour < 20) & (data.index.hour > 5), 'Strategy Returns'] = 0

    #Set leverage
    data['Strategy Returns'] = data['Strategy Returns'] * leverage

    # Calculate cumulative returns
    data['Cumulative Returns'] = (1 + data['Strategy Returns']).cumprod()
    
    
    final_equity = data['Cumulative Returns'].iloc[-1] * initial_capital
    

    #import matplotlib.pyplot as plt
    #data.dropna(inplace=True)   
    #plt.plot(data['Cumulative Returns'], label='Strategy Returns')
    #plt.show()

    if return_series == True:
        return data['Cumulative Returns']
    else:
        return final_equity



