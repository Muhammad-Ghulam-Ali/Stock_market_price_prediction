import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

# Load the Training Dataset
dataset_train = pd.read_csv("<path to downloaded Training dataset.csv>")
dataset_train.head()

# Using Open Stock Price Column to Train the Model
training_set = dataset_train.iloc[:,1:2].values

# Normalizing the Dataset
scaler = MinMaxScaler(feature_range= (0, 1))
scaled_training_set = scaler.fit_transform(training_set)

# Creating x_train, y_train Data Structures
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(scaled_training_set[i - 60: i, 0])
    y_train.append(scaled_training_set[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Reshaping the Data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Adding Different Layers to the LSTM 
regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

# Fitting the Model
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train, y_train, epochs=250, batch_size=32)

# Getting the Actual Stock Prices
dataset_test = pd.read_csv("<path to actual stock dataset.csv>")
actual_stock_price = dataset_test.iloc[:,1:2].values

# Preparing the Input for the Model
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values

inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

x_test = []
for i in range(60, 124):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predicting the Stock Price
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Plotting the Actual vs Predicted Graph
plt.plot(actual_stock_price, color = 'red', label = 'Actual Google Stock  Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()

plt.show()