import pandas as pd
import numpy as np
from scipy import stats 
import scipy.optimize
from scipy.optimize import OptimizeWarning
import warnings
import math 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# from mplfinance  import _candlestick
from matplotlib.dates import date2num
from datetime import datetime

from sklearn.utils.validation import FLOAT_DTYPES

class Holder:
    1


def heikenashi(prices, periods):

    """
    :param prices: dataframe of OHLC & volume data
    :param periods: periods for which to create the candles
    :return: heiken ashi OHLC candles
    """

    results = Holder()

    dict = {}

    HAclose = prices[['open','high','low','close']].sum(axis=1)/4
    
    HAopen = HAclose.copy()   
    HAopen.iloc[0] = HAclose.iloc[0]

    HAhigh = HAclose.copy()   
    HAlow = HAclose.copy()   

    for i in range(1,len(prices)):
        HAopen.iloc[i] = (HAopen.iloc[i-1]+HAclose.iloc[i-1])/2
        HAhigh.iloc[i] = np.array([prices.high.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).max()
        HAhigh.iloc[i] = np.array([prices.low.iloc[i], HAopen.iloc[i], HAclose.iloc[i]]).min()

    df = pd.concat((HAopen,HAhigh,HAlow,HAclose), axis=1)
    df.columns = [['open','high','low','close']]

    # df.index = df.index.droplevel(0)

    dict[periods[0]] = df
    results.candles = dict

    return results

def detrend(prices, method='difference'):

    """
    :param prices: dataframe of OHLC
    :param method: linear or difference
    :return: the detrended price series
    """

    if method == 'difference':
        detrended = prices.close[1:]-prices.close[:-1].values
    
    elif method == 'linear':
        x = np.arange(0,len(prices))
        y = prices.close.values

        model = LinearRegression()
        model.fit(x.reshape(-1,1),y.reshape(-1,1))

        trend = model.predict(x,reshape(-1,1))

        trend = trend.reshape((len(prices),))

        detrended = prices.close - trend
    else:
        print('Invalid method for detrending')

    return detrended
        
def fseries(x, a0, a1, b1, w):
    """
    :param x: the hours
    :param a0: coefficient
    :param a1: coefficient
    :param b1: coefficient
    :param w: frequency
    :return: the value of fourier function
    """

    f = a0 + a1*np.cos(w*x) + b1*np.sin(w*x)

    return f

def sseries(x, a0, b1, w):
    """
    :param x: the hours
    :param a0: coefficient
    :param b1: coefficient
    :param w: frequency
    :return: the value of fourier function
    """

    f = a0 + b1*np.sin(w*x)

    return f

def fourier(prices, periods, method='difference'):
    """
    :param prices: dataframe of OHLC
    :param periods: periods for which to compute coefficients
    :param method: linear or difference
    :return: dict of dataframes
    """

    results = Holder()
    dict = ()

    plot = False

    detrended = detrend(prices, method)

    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)-periods[i]):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(fseries,x,y)
                except(RuntimeError, OptimizeWarning):
                    res =np.empty((1,4))
                    res[0:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = fseries(xt, res[0][0], res[0][1], res[0][2], res[0][3])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs.reshape((len(coeffs)/4, 4)))
        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])
        df.columns = [['a0', 'a1', 'b1', 'w']]
        df =df.fillna(method='bfill')
        dict[periods[i]] = df
    results.coeffs = dict

    return results

def sine(prices, periods, method='difference'):
    """
    :param prices: dataframe of OHLC
    :param periods: periods for which to compute coefficients
    :param method: linear or difference
    :return: dict of dataframes
    """

    results = Holder()
    dict = ()

    plot = False

    detrended = detrend(prices, method)

    for i in range(0, len(periods)):
        coeffs = []

        for j in range(periods[i], len(prices)-periods[i]):
            x = np.arange(0,periods[i])
            y = detrended.iloc[j-periods[i]:j]

            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)

                try:
                    res = scipy.optimize.curve_fit(sseries,x,y)
                except(RuntimeError, OptimizeWarning):
                    res =np.empty((1,3))
                    res[0:] = np.NAN

            if plot == True:
                xt = np.linspace(0, periods[i], 100)
                yt = sseries(xt, res[0][0], res[0][1], res[0][2])

                plt.plot(x, y)
                plt.plot(xt, yt, 'r')
                plt.show()

            coeffs = np.append(coeffs, res[0], axis=0)

        warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
        
        coeffs = np.array(coeffs.reshape((len(coeffs)/3, 3)))
        df = pd.DataFrame(coeffs, index=prices.iloc[periods[i]:-periods[i]])
        df.columns = [['a0', 'b1', 'w']]
        df =df.fillna(method='bfill')
        dict[periods[i]] = df
    results.coeffs = dict

    return results



