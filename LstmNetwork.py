#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 08:33:31 2017

Adopted from:
    http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Guide to Keras:
    https://keras.io/getting-started/sequential-model-guide/#training

@author: valentin
"""

# Import the functions and classes we'll need
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from pydataset import data
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# read in the Boston housing data
dataframe = data("Boston")

# select only specific columns - MAKE SURE THAT THE TARGET IS IN the first column
dataframe = dataframe[['medv', 'crim', 'rm', 'age', 'lstat']]
xVarCols = [1, 2, 3, 4]
yVarCols = [0]


# fix random seed for reproducibility
numpy.random.seed(7)

# Extract the NumPy array from the dataframe and convert the integer values to
# floating point values, which are more suitable for modeling with a neural network
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalize the data between 0 - 1
scaler = MinMaxScaler(feature_range=(0, 1), copy = True)
dataset = scaler.fit_transform(dataset)

# Split the data in train and validaiton
trainSize = int(len(dataset) * 2/3)

train = dataset[0:trainSize, :]
test = dataset[trainSize:len(dataset), :]

print(len(train), len(test), len(dataset))


## Modify the data for the LSTM network - The LSTM network expects the input data (X)
# to be provided with a specific array structure in the form of: [samples, time steps, features].
trainY = train[ :, yVarCols]
trainX = train[ :, xVarCols]

testX = test[ :, xVarCols]

dataframe_length = len(trainY)
# dataframe_dim = Need to figure out how to count the columns of the array


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX =  numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

## The LSTM network expects the input data (X) to be provided with a specific
# array structure in the form of: [samples, time steps, features]
# Define the network
modelFit = Sequential()
modelFit.add(LSTM(4,
                  activation = 'sigmoid',
                  input_shape = (1, 4)))
modelFit.add(Dense(1))

# Before training the model, configure the learning process via the compile method
modelFit.compile(optimizer = 'rmsprop',
                 loss = 'mean_squared_error',
                 metrics = ['accuracy'])

# Train the model
modelFit.fit(trainX, trainY,
             epochs = 5,
             batch_size = 1,
             verbose = 1)

# make predictions
trainPredict = modelFit.predict(trainX)
testPredict = modelFit.predict(testX)



##### WORK FROM HERE ON - THE CODE IS DONE THROUGH HERE !!!


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# Plot the forecast and actuals
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()



