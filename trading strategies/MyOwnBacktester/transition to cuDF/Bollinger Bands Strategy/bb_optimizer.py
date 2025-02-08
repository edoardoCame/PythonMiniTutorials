

#Le combinazioni le devi importare dall'esterno
#my_combinations = pd.read_csv('/home/edoardocame/Desktop/python_dir/PythonMiniTutorials/trading strategies/MyOwnBacktester/transition to cuDF/my_combinations.csv', index_col=0)



def bb_optimizer(my_combinations, pair):
    import time as t
    from bollinger_filter import backtest_bollinger_strategy
    start_time = t.time()
    ####################
    my_combinations['result'] = 0
    def apply_strategy(row):
        result = backtest_bollinger_strategy(data=pair, lookback=row['lookback'], 
                                             sdev=row['sdev'], leverage=10, filter=row['filter'], fee_percentage=0.01)
        print(f"Processed lookback: {row['lookback']}, sdev: {row['sdev']}, result: {result}")
        return result

    my_combinations['result'] = my_combinations.apply(apply_strategy, axis=1)
    ####################

    end_time = t.time() - start_time

    print(f"Time elapsed: {end_time}")
    return my_combinations
