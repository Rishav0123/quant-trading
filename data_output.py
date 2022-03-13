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

df = pd.read_csv("data_input_x.csv")
#print(df)
df = df["Close"]
#print(price)
df = df.iloc[1:]
#price_y = pd.DataFrame({'price_output' : price.values})
#price_y['price_output'] = price_y['price_output'].shift(1)
print(df)
print(df.shape)
df.to_csv('data_output.csv')
