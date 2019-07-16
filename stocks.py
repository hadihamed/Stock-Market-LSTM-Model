# -*- coding: utf-8 -*-
"""
This program loads stock market data from AlphaVantage, trains an RNN
through converting stock prices into time-series data, and outputs a
predicted stock price.

@author Hadi Hamed 
"""

#import basic libraries and Alpha Vantage stock market data API
import numpy as np
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries

# Neural Nets
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler
import time

#Downloading last 100 days of aapl stock prices
ts = TimeSeries(key = 'QL1EZRMC4W85QQDE', output_format = 'pandas')
raw_data, meta_data = ts.get_daily_adjusted('AAPL', outputsize = 'compact')

#taking only opening price for univariate analysis
data = raw_data[['1. open']]

#Renaming input data column
data.columns = ['open']

# Normalizing data into 0-1 range
scaler = MinMaxScaler(feature_range = (0,1))
data['open'] = scaler.fit_transform(data)

#Original stock price plot
plt.plot(data.index, data['open'])
plt.title('AAPL Stock Open')
plt.ylabel('Price')
plt.show()

#Empty lists for training features and labels as time-series variables 
features_set = []
labels = []

model_days = 40 #Number of days which the price is given as an input variable
test_days = 20 #Number of days for testing the trained model

'''Builds training set time-series table by taking the prior model_days as
 inputs for every day between the minimum possible start days and
 the test days'''
for i in range(model_days, data.shape[0] - test_days):
    features_set.append(data['open'][i-model_days: i])
    labels.append(data['open'][i])
    
#Create empty lists for test features and labels     
test_features = []
test_labels = []

'''Builds test set time-series table by taking the prior model_days as
 inputs for the most recent test_days'''
for i in range(data.shape[0] - test_days, data.shape[0]):
    test_features.append(data['open'][i-model_days: i])
    test_labels.append(data['open'][i])
    
#Converting lists into numpy arrays
features_set, labels = np.array(features_set), np.array(labels)
test_features, test_labels = np.array(test_features), np.array(test_labels)

#reshaping training and test sets into 3-d arrays
features_set = np.reshape(features_set, (features_set.shape[0],
                                         features_set.shape[1], 
                                         1))

test_features = np.reshape(test_features, (test_features.shape[0],
                                                   test_features.shape[1],
                                                   1))

#Actual prices for test set
actual_test_price = scaler.inverse_transform(
        data.iloc[(data.shape[0] - test_days):])

#Building Sequential Keras model
model = Sequential()

model.add(LSTM(units = 50,
               return_sequences = True,
               input_shape = (features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM( units = 100,
               return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM( units = 50,
               return_sequences = False))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

#Saving start time to calculate total model runtime
start = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
print ('compilation time : ', time.time() - start)

model.fit(
        features_set,
        labels,
        epochs = 10,
        batch_size = 10)

#Make predictions and re-scale output to currency value
predictions = model.predict(test_features)
predictions = scaler.inverse_transform(predictions)

#Plotting predictions against actual stock price for test period
plt.figure(figsize = (10, 6))
plt.plot(actual_test_price, color = 'blue', label = 'Actual Stock Price')
plt.plot(predictions, color = 'red', label = 'Predicted Stock Price')
plt.title('Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Apple Stock Price')
plt.legend()
plt.show()



