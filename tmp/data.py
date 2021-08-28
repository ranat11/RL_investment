import pandas as pd
import plotly as py
import plotly.graph_objs as go
from feature_functions import *

## Load data
df = pd.read_csv('data/EURUSDhours.csv')
df.columns = ['date','open','high','low','close','volume']

df.date = pd.to_datetime(df.date, format='%d.%m.%Y %H:%M:%S.%f')
df = df.set_index(df.date)
df = df[['open','high','low','close','volume']]
df = df.drop_duplicates(keep=False)
df = df.iloc[:200]
print(df.head())
exit()

## Moving average
ma = df.close.rolling(center=False,window=30).mean()


## get function data
HAresults = heikenashi(df, [1])
HA = HAresults.candles[1]

detrended = detrend(df, method='difference')

f = sine(df, [10,15], method='difference')
exit()

## Plot graph
trace0 = go.Ohlc(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close, name='Currency Quote')
trace1 = go.Scatter(x=df.index, y=ma)
trace2 = go.Scatter(x=df.index, y=detrended)
# trace2 = go.Ohlc(x=HA.index, open=HA.open, high=HA.high, low=HA.low, close=HA.close, name='heiken Ashi')
# trace2 = go.Bar(x=df.index, y=df.volume)

data = [trace0,trace1,trace2]

fig = py.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True)
fig.append_trace(trace0,1,1)
fig.append_trace(trace1,1,1)
fig.append_trace(trace2,2,1)

py.offline.plot(fig,filename='data_plot.html') 