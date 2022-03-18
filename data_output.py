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

df = pd.read_csv("data_input_raw.csv")
output = pd.read_csv("data_output.csv")
#print(df)
#flag = 1
df1 = df.loc[:,'Close']
df1 = df1.iloc[-1:]

#df1 = df.loc[-1:'Close']
print("*****")
print(df1)
output = output.append(df1)





#price_y = pd.DataFrame({'price_output' : price.values})
#price_y['price_output'] = price_y['price_output'].shift(1)
#print(df)
print("Y size: ")
#print(df.shape)
output.to_csv('data_output.csv')
