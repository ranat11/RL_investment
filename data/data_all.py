import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from matplotlib import pyplot as plt
import mplfinance as mpf
from finta import TA

# https://www.dukascopy.com/swiss/english/marketwatch/historical/ #UTC local
# https://www.alphavantage.co/

file_name = "EURUSDdays"

df = pd.read_csv(f'data/{file_name}.csv')
df.columns = ['date','open','high','low','close','volume']

# remove GMT 
# for i in range (len(df)):
#     df['date'].values[i] = df['date'].values[i][:-9]
    
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M:%S.%f', infer_datetime_format=True)
df.set_index('date', inplace=True)
df = df.drop_duplicates(keep=False)

# ---------------------------------------------------------------------------- #
#                                Add day of week                               #
# ---------------------------------------------------------------------------- #
df["dayofweek"] = df.index.dayofweek
df["date"] = df.index.day
df["month"] = df.index.month

## For hours
df["hours"] = df.index.hour

# ---------------------------------------------------------------------------- #
#                          Create technical indicators                         #
# ---------------------------------------------------------------------------- #
def create_ti(name, cal_date):
    for i in cal_date:
        df[f'{name}{i}'] = getattr(TA,name)(df, i)
    return

create_ti('SMA', [3,4,5,8,9,10,20,60,120])
create_ti('STOCHD', [3,4,5,8,9,10,20,60,120])
create_ti('RSI', [7,14,30])
create_ti('WILLIAMS', [6,7,8,9,10])
create_ti('CCI', [15])

df['OBV'] = TA.OBV(df)

# TODO with concat
# df['MACD15'],df['MACD_signal15'] = TA.MACD(df,15)
# df['MACD30'],df['MACD_signal30'] = TA.MACD(df,30)
# df['upper_bb'],df['middle_band'],df['lower_bb'] = TA.BBANDS(df,15)

df.dropna(inplace = True)
df.to_csv(f'data/{file_name}_ti.csv')


# ---------------------------------------------------------------------------- #
#                                  Correlation                                 #
# ---------------------------------------------------------------------------- #
corr_matrix = df.corr()
print(corr_matrix["close"].sort_values(ascending=False))

attributes = ["close", "SMA4", "STOCHD120", "RSI30"]
img = scatter_matrix(df[attributes], figsize=(12, 8))
plt.show()
# ---------------------------------------------------------------------------- #
#                                   Normalize                                  #
# ---------------------------------------------------------------------------- #
print(df.describe())
exit()
# df = (df-df.min())/(df.max()-df.min())



# ---------------------------------------------------------------------------- #
#                                   plot data                                  #
# ---------------------------------------------------------------------------- #
# print(df.head(20))
print(df.info())
# print(df.tail(50))
# import mplfinance as mpf
# mpf.plot(df, type='candle', mav = (3,5,10), volume = True)
