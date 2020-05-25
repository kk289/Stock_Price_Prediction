# Stock Price Prediction Model for Netfliximport pandas as pd

# Importing Libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader.data as web
import datetime
import seaborn as sb
import os

images_folder = "images"

seed = 0

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
plt.savefig(os.path.join(images_folder, "1_corr_visual.png"))

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
#plt.show()
plt.savefig(os.path.join(images_folder, "2_priceHistory.png"))

# Plot Open vs Close (Year 2012)
appl_df[['Open','Close']].head(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "3_openVScloseYear2012.png"))

# Plot Open vs Close (Year 2020)
appl_df[['Open','Close']].tail(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "4_openVScloseYear2020.png"))

# Plot High vs Close (Year 2012)
appl_df[['High','Close']].head(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "5_highVScloseYear2012.png"))

# Plot High vs Close (Year 2020)
appl_df[['High','Close']].tail(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "6_highVScloseYear2020.png"))

# Plot Low vs Close (Year 2012)
appl_df[['Low','Close']].head(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "7_lowVScloseYear2012.png"))

# Plot Low vs Close (Year 2020)
appl_df[['Low','Close']].tail(50).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "8_lowVScloseYear2020.png"))

if __name__ == '__main__':
    np.random.seed(seed)

# Model Training and Testing

# Date format is DateTime 
appl_df['Year'] = df['Date'].dt.year
appl_df['Month'] = df['Date'].dt.month
appl_df['Day'] = df['Date'].dt.day

# final dataset for model training
final_appl = appl_df[['Day', 'Month', 'Year', 'High', 'Open', 'Low', 'Close']]
print(final_appl.head(10))
print(final_appl.tail(10))

#separate Independent and dependent variable
X = final_appl.iloc[:,final_appl.columns != 'Close']
Y = final_appl.iloc[:, 5]
print(X.shape)  #output: (2111, 6)
print(Y.shape)  #output: (2111,)

# Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(X,Y,test_size=.25)
print(x_train.shape) #output: (1583, 6)
print(x_test.shape)  #output: (528, 6)  
print(y_train.shape) #output: (1583,)
print(y_test.shape)  #output: (528,)
#y_test to be evaluated with y_pred for Diff models

## Model 1: Linear Regression Model

# Linear Regression Model Training and Testing
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
y_pred = lr_model.predict(x_test)

# Linear Model Cross-Validation
from sklearn import model_selection
from sklearn.model_selection import KFold
kfold = model_selection.KFold(n_splits=20, random_state=seed, shuffle=True)
results_kfold = model_selection.cross_val_score(lr_model, x_test, y_test.astype('int'), cv=kfold)
print("Accuracy: ", results_kfold.mean()*100)
# Accuracy: 99.99743780203187

# Plot Actual vs Predicted Value
plot_df = pd.DataFrame({'Actual':y_test,'Pred':y_pred})
plot_df.head(10).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "9_actualVSpredictedLRM.png"))

## Model 2: KNN: K-nearest neighbor Regression Model

# KNN Model Training and Testing
from sklearn.neighbors import KNeighborsRegressor
knn_regressor = KNeighborsRegressor(n_neighbors = 4)
knn_model = knn_regressor.fit(x_train,y_train)
y_knn_pred = knn_model.predict(x_test)

# KNN Cross-Validation
knn_kfold = model_selection.KFold(n_splits=20, random_state=seed, shuffle=True)
results_kfold = model_selection.cross_val_score(knn_model, x_test, y_test.astype('int'), cv=knn_kfold)
print("Accuracy: ", results_kfold.mean()*100)
# Accuracy: 99.91435220285842

# Plot Actual vs Predicted
plot_knn_df = pd.DataFrame({'Actual':y_test,'Pred':y_knn_pred})
plot_knn_df.head(10).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "10_actualVSpredictedkNN.png"))

# Model 3: SVM Support Vector Machine Regression Model

# SVM Model Training and Testing
from sklearn.svm import SVR
svm_regressor = SVR(kernel='linear')
svm_model = svm_regressor.fit(x_train,y_train)
y_svm_pred = svm_model.predict(x_test)

# SVM Cross-Validation
svm_kfold = model_selection.KFold(n_splits=20, random_state=seed, shuffle=True)
results_kfold = model_selection.cross_val_score(svm_model, x_test, y_test.astype('int'), cv=svm_kfold)
print("Accuracy: ", results_kfold.mean()*100)
# Accuracy: 99.99301338392715

# Plot Actual vs Predicted
plot_svm_df = pd.DataFrame({'Actual':y_test,'Pred':y_svm_pred})
plot_svm_df.head(10).plot(kind='bar',figsize=(16,8))
plt.grid(which ='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which ='minor', linestyle=':', linewidth='0.5', color='black')
#plt.show()
plt.savefig(os.path.join(images_folder, "11_actualVSpredictedSVM.png"))

# RMSE (Root Mean Square Error)
from sklearn.metrics import mean_squared_error , r2_score
import math
lr_mse = math.sqrt(mean_squared_error(y_test,y_pred))
print('Linear Model Root mean square error', lr_mse)

knn_mse = math.sqrt(mean_squared_error(y_test,y_knn_pred))
print('KNN Model Root mean square error', knn_mse)

svm_mse = math.sqrt(mean_squared_error(y_test,y_svm_pred))
print('SVM Model Root mean square error', svm_mse)

# R-squared Error
print('Linear R2: ', r2_score(y_test, y_pred))
print('KNN R2: ', r2_score(y_test, y_knn_pred))
print('SVM R2: ', r2_score(y_test, y_svm_pred))