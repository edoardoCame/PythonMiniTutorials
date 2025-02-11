

##############################################################################################################
#Here I define all the backtesting functions:
##############################################################################################################



def backtest_bollinger_strategy(data, initial_capital=100, fee_percentage=0.005, 
                                lookback=60, sdev=5, return_series=False, leverage=1):
    
    import cupy as cp

    lookback = int(lookback)
    sdev = int(sdev)
    #Here I calculate the bollinger bands first:
    rolling_mean = data['close'].rolling(window=lookback).mean()
    rolling_std = data['close'].rolling(window=lookback).std()
    data['Middle Band'] = rolling_mean
    data['Upper Band'] = rolling_mean + (rolling_std * sdev)
    data['Lower Band'] = rolling_mean - (rolling_std * sdev)


    data['Signal'] = cp.nan #I use NA to ffill the positions


    data.loc[data['close'] < data['Lower Band'], 'Signal'] = 1
    data.loc[data['close'] > data['Upper Band'], 'Signal'] = -1
    
    # Calculate position changes based on middle band crosses
    data['Above_Middle'] = data['close'] > data['Middle Band']

    data['Middle_Cross_Up'] = (data['Above_Middle'] != data['Above_Middle'].shift(1)) & data['Above_Middle']
    
    data['Middle_Cross_Down'] = (data['Above_Middle'] != data['Above_Middle'].shift(1)) & (~data['Above_Middle']) #~ is NOT operator
    
    

    # Initialize positions
    data['Position'] = data['Signal']
    
    # Update positions based on middle band crosses
    long_exits = data['Middle_Cross_Down']
    short_exits = data['Middle_Cross_Up']
    
    # Apply position updates
    data.loc[long_exits | short_exits, 'Position'] = 0 #when long/short exits trigger, position is reset to zero;
    
    # Forward fill positions
    data['Position'] = data['Position'].fillna(method='ffill')
    data['Position'] = data['Position'].fillna(0).astype('float64')
    
    # Calculate returns
    data['Returns'] = data['close'].pct_change()
    data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']



    # Identify when an order is triggered (when the position changes)
    data['Order Triggered'] = data['Position'] != data['Position'].shift(1)

    # Only apply fee when entering a position (non-zero current position) THIS WAY YOU ENTER ONLY ONCE!
    data.loc[data['Order Triggered'] & (data['Position'] != 0), 'Strategy Returns'] -= abs(fee_percentage / 100)
    


    # Set leverage
    data['Strategy Returns'] = data['Strategy Returns'] * leverage

    # Calculate cumulative returns
    data['Cumulative Returns'] = (1 + data['Strategy Returns']).cumprod()
    data['Buy & Hold Returns'] = (1 + data['Returns']).cumprod()
    
    final_equity = data['Cumulative Returns'].iloc[-1] * initial_capital
    
    if return_series == True:
        return data['Cumulative Returns']
        #return data
    else:
        return final_equity

