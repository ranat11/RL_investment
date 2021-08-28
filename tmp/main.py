
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import mplfinance as mpf
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# start = datetime.datetime(2015, 1, 1)
# end = datetime.datetime.now()

# df = web.DataReader("TSLA", 'yahoo', start, end)
# df.to_csv('data/tsla.csv')

df = pd.read_csv('data/tsla.csv', parse_dates=True, index_col=0)
df.index.name = 'Date'
df['100ma'] = df['Close'].rolling(window=100, min_periods=0).mean()


mpf.plot(df, type='candle', mav = (3,5,10,100), volume = True)

# ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
# ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
# ax1.plot(df.index, df['Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])

# plt.show()
