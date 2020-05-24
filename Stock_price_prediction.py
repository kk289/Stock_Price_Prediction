# Stock Price Prediction Model for Netfliximport pandas as pd

# Importing Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import datetime
# pandas_datareader library allows us to connect to the website and extract data directly from internet sources in our case we are extracting data from Yahoo Finance API.

start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2020, 5, 22)

# Dataset review
df = web.DataReader("AAPL", 'yahoo', start, end)

dates =[]
for x in range(len(df)):
    newdate = str(df.index[x])
    newdate = newdate[0:10]
    dates.append(newdate)
df['dates'] = dates

print(df.tail(10))
