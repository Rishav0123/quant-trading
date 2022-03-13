import numpy as np
import pandas as pd
#Data Source
import yfinance as yf
#Data viz
import plotly.graph_objs as go
import requests
import pandas as pd
import numpy as np
from math import floor
from termcolor import colored as cl
import matplotlib.pyplot as plt
import os
from subprocess import call
import time

plt.rcParams['figure.figsize'] = (20, 10)
plt.style.use('fivethirtyeight')


#train_macd


data_train = yf.download(tickers='ETH-USD', period = '1m', interval = '7d')
print(data_train)
def get_macd(price, slow, fast, smooth):
    exp1 = price.ewm(span = fast, adjust = False).mean()
    exp2 = price.ewm(span = slow, adjust = False).mean()
    macd = pd.DataFrame(exp1 - exp2).rename(columns = {'Close':'macd'})
    signal = pd.DataFrame(macd.ewm(span = smooth, adjust = False).mean()).rename(columns = {'macd':'signal'})
    hist = pd.DataFrame(macd['macd'] - signal['signal']).rename(columns = {0:'hist'})
    frames =  [macd, signal, hist]
    data_train = pd.concat(frames, join = 'inner', axis = 1)
    return data_train

data_macd = get_macd(data_train['Close'], 26, 12, 9)
#print(data_macd.head())

#print(data_macd.shape[1])



#train_2050_SME

# create 20 days simple moving average column
data_train['20_SMA'] = data_train['Close'].rolling(window = 20, min_periods = 1).mean()
# create 50 days simple moving average column
data_train['50_SMA'] = data_train['Close'].rolling(window = 50, min_periods = 1).mean()
# display first few rows
print(data_train.head())

#train_bollinger_band

data_train['TP'] = (data_train['Close'] + data_train['Low'] + data_train['High'])/3
data_train['std'] = data_train['TP'].rolling(20).std(ddof=0)
data_train['MA-TP'] = data_train['TP'].rolling(20).mean()
data_train['BOLU'] = data_train['MA-TP'] + 2*data_train['std']
data_train['BOLD'] = data_train['MA-TP'] - 2*data_train['std']
#print(data_train)
#print(data_train.shape[1])

#Super trend calculation

def Supertrend(data_train, atr_period, multiplier):


        high = data_next['High']
        low = data_next['Low']
        close = data_next['Close']

         #calculate ATR
        price_diffs = [high - low,
                       high - close.shift(),
                       close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean()
        #df['atr'] = df['tr'].rolling(atr_period).mean()

        # HL2 is simply the average of high and low prices
        hl2 = (high + low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (multiplier * atr)
        final_lowerband = lowerband = hl2 - (multiplier * atr)

        # initialize Supertrend column to True
        supertrend = [True] * len(data_train)

        for i in range(1, len(data_train.index)):
            curr, prev = i, i-1

            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan

        return pd.DataFrame({
            'Supertrend': supertrend,
            'Final Lowerband': final_lowerband,
            'Final Upperband': final_upperband
        }, index=data_train.index)



atr_period = 10
atr_multiplier = 3.0

supertrend = Supertrend(data_train, atr_period, atr_multiplier)
data_train = data_train.join(supertrend)

print(data_train.shape[1])

for cols in data_train.columns:
    print(cols)
#SUPERTREND
#print(data_train)
data_train.to_csv('data_input_x.csv')
#os.system('data_output.py')
call('python data_output.py', shell=True)
#F_price = data_train.iloc[-1, data_train.columns.get_loc("Close")]
#print(data_train.drop(data_train.tail(1).index,inplace=True))
print(data_train.shape)
print(data_train)
data_next = data_train.iloc[-1:]
data_train = data_train.iloc[:-1]

data_next.to_csv('data_next.csv')
#print(data_train)
print(data_train)
data_train.to_csv('data_input.csv')
#print(F_price)
#data = data_train[-1]
#print(data_train)
call('python train.py' , shell=True)
call('python plot.py', shell=True)

flag = 1

#while flag ==1:
#    print('waiting...')
#    print(data_next)
#    data_macd = get_macd(data_next['Close'], 26, 12, 9)
    #print(data_next['Close'])
    # create 20 days simple moving average column
#    data_next['20_SMA'] = data_next['Close'].rolling(window = 20, min_periods = 1).mean()
    # create 50 days simple moving average column
#    data_next['50_SMA'] = data_next['Close'].rolling(window = 50, min_periods = 1).mean()

#    data_next['TP'] = (data_next['Close'] + data_next['Low'] + data_next['High'])/3
#    data_next['std'] = data_next['TP'].rolling(20).std(ddof=0)
#    data_next['MA-TP'] = data_next['TP'].rolling(20).mean()
#    data_next['BOLU'] = data_next['MA-TP'] + 2*data_next['std']
#    data_next['BOLD'] = data_next['MA-TP'] - 2*data_next['std']

#    def Supertrend(data_next, atr_period, multiplier):

#        high = data_next['High']
#        low = data_next['Low']
#        close = data_next['Close']

        # calculate ATR
#        price_diffs = [high - low,
#                       high - close.shift(),
#                       close.shift() - low]
#        true_range = pd.concat(price_diffs, axis=1)
#        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
#        atr = true_range.ewm(alpha=1/atr_period,min_periods=atr_period).mean()
        # df['atr'] = df['tr'].rolling(atr_period).mean()

        # HL2 is simply the average of high and low prices
#        hl2 = (high + low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
#        final_upperband = upperband = hl2 + (multiplier * atr)
#        final_lowerband = lowerband = hl2 - (multiplier * atr)

        # initialize Supertrend column to True
#        supertrend = [True] * len(data_train)

#        for i in range(1, len(data_train.index)):
#            curr, prev = i, i-1

            # if current close price crosses above upperband
#            if close[curr] > final_upperband[prev]:
#                supertrend[curr] = True
            # if current close price crosses below lowerband
#            elif close[curr] < final_lowerband[prev]:
#                supertrend[curr] = False
            # else, the trend continues
#            else:
#                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
#                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
#                    final_lowerband[curr] = final_lowerband[prev]
#                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
#                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
#            if supertrend[curr] == True:
#                final_upperband[curr] = np.nan
#            else:
#                final_lowerband[curr] = np.nan

#        return pd.DataFrame({
#            'Supertrend': supertrend,
#            'Final Lowerband': final_lowerband,
#            'Final Upperband': final_upperband
#        }, index=data_train.index)


#    atr_period = 10
#    atr_multiplier = 3.0

#    supertrend = Supertrend(data_next, atr_period, atr_multiplier)
#    data_next = data_next.join(supertrend)
#    print(data_next)
