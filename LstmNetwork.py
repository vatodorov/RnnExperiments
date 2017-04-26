#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 08:33:31 2017

This is a trained Recurrent Neural Network (LSTM) to predict time series data

Adopted from:
    http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Guide to Keras:
    https://keras.io/getting-started/sequential-model-guide/#training

@author: valentin
"""

# Import the functions and classes we'll need
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# read in the Boston housing data
df = datasets.load_boston()
df2 = df["data"]


# select only specific columns - MAKE SURE THAT THE TARGET IS IN the first column
dataframe = df2[:, (11, 0, 6, 12, 5), ]

xVarCols = [1, 2, 3, 4]
yVarCols = [0]


# fix random seed for reproducibility
np.random.seed(7)

## NO NEED FOR ARRAYS - Extract the NumPy array from the dataframe and convert the integer values to
# floating point values, which are more suitable for modeling with a neural network
# dataset = dataframe.values
# dataset = dataset.astype('float32')

# Normalize the data between 0 - 1
scaler = MinMaxScaler(feature_range = (0, 1), copy = True)
dataset = scaler.fit_transform(dataframe)

# Split the data in train and validaiton
trainSize = int(len(dataset) * 2/3)

train = dataset[0:trainSize, :]
validate = dataset[trainSize:len(dataset), :]

print(len(train), len(validate), len(dataset))


## Modify the data for the LSTM network - The LSTM network expects the input data (X)
# to be provided with a specific array structure in the form of: [samples, time steps, features].
trainX = train[ :, xVarCols]
validateX = validate[ :, xVarCols]

trainY = train[ :, yVarCols]
validateY = validate[ :, yVarCols]

dataframe_length = len(trainY)
# dataframe_dim = Need to figure out how to count the columns of the array

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
validateX =  np.reshape(validateX, (validateX.shape[0], 1, validateX.shape[1]))



## The LSTM network expects the input data (X) to be provided with a specific
# array structure in the form of: [samples, time steps, features]
# Define the network
modelFit = Sequential()
modelFit.add(LSTM(4,
                  activation = 'sigmoid',
                  input_shape = (1, 4)))
modelFit.add(Dense(1))

# Before training the model, configure the learning process via the compile method
modelFit.compile(optimizer = 'adam',
                 loss = 'mean_squared_error',
                 metrics = ['accuracy'])

# Train the model
modelEstimate = modelFit.fit(trainX, trainY,
                             epochs = 5,
                             batch_size = 1,
                             verbose = 1,
                             validation_data=(validateX, validateY))

# make predictions
trainPredict = modelFit.predict(trainX)
validatePredict = modelFit.predict(validateX)

# print the training accuracy and validation loss at each epoch
print(modelEstimate.history)

# print the number of models of the network
print(len(modelFit.layers))


# Invert predictions
# Combine the arrays - stack the forecast with the predictive features
df_train = np.column_stack((trainPredict, train[:, 1:]))
trainPredict2 = scaler.inverse_transform(df_train)

df_validate = np.column_stack((validatePredict, validate[:, 1:]))
validatePredict2 = scaler.inverse_transform(df_validate)

# calculate root mean squared error



##### Re-write the code from here on

# Plot the forecast and actuals
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Keras RNN Error History')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epoch')
plt.legend(['Training Error', 'Validation Error'], loc='upper left')
plt.show()

# Plotting the hypothesis and Cross Validation y
plt.plot(validationY[:-predictionPeriod])
plt.plot(ho[predictionPeriod:]+np.mean(validationY)-np.mean(ho))
plt.title('Keras Prediction Cross Validation Plot')
plt.legend(['Hypothesis', 'Actual'], loc='upper left')
plt.show()


