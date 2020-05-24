# Stock Price Prediction Model for Netfliximport pandas as pd

# Importing Libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import datetime
import seaborn as sb
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

print(df.head(10))
print(df.tail(10))

df.reset_index(inplace=True)
print(df.describe())

df.describe(include=np.object)

# Pearson Correlation Coefficient
corr = df.corr(method='pearson')
print(corr)

# Visulize correlation
corr_visual = sb.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,cmap='RdBu_r', annot=True, linewidth=0.5)
plt.savefig("corr_visual.png")

# Visualize the Dependent variable with Independent Features
#prepare dataset to work with 
appl_df = df[['Date','High','Open','Low','Close']]
print(appl_df.head(20))
plt.figure(figsize=(16,8))
plt.title('Apple Stocks Closing Price History 2012-2020')
plt.plot(appl_df['Date'],appl_df['Close'])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price US($)',fontsize=18)
plt.style.use('fivethirtyeight')
plt.show()

# Plot Open vs Close (Year 2012)
appl_df[['Open','Close']].head(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Plot Open vs Close (Year 2020)
appl_df[['Open','Close']].tail(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Plot High vs Close (Year 2012)
appl_df[['High','Close']].head(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# Plot High vs Close (Year 2020)
appl_df[['High','Close']].tail(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()







