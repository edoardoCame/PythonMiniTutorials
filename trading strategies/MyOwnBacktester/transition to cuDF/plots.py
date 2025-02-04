import pandas as pd
import plotly.graph_objects as go

my_combinations = pd.read_csv('/home/edoardocame/Desktop/python_dir/PythonMiniTutorials/trading strategies/MyOwnBacktester/transition to cuDF/my_combinations_results.csv')


z = my_combinations['result'].values
x = my_combinations['sdev'].values
y = my_combinations['lookback'].values

fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=12, color=z, colorscale='Viridis', opacity=1))])

fig.update_layout(
    title='result data set',
    scene=dict(
        xaxis_title='Standard Deviation',
        yaxis_title='Lookback',
        zaxis_title='Result'
    )
)
fig.show()