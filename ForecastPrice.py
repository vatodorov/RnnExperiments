
"""
This is a trained Recurrent Neural Network (LSTM) to predict time series data

Adopted from:
    http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

Guide to Keras:
    https://keras.io/getting-started/sequential-model-guide/#training

@author: valentin
"""

## Import the functions and classes we'll need
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import backend as K

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


## Define model
def rnnModel(inputData, selectColumns, subsetXVarColumns, subsetYVarColumns, randSeed):
             
    # susbser the data
    dataframe = inputData[:, (selectColumns), ]
    xVarColumns = subsetXVarColumns
    yVarColumns = subsetYVarColumns
    
    # fix random seed for reproducibility
    np.random.seed(randSeed)
    
    
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
    trainX = train[:, xVarColumns]
    validateX = validate[:, xVarColumns]
    
    trainY = train[:, yVarColumns]
    validateY = validate[:, yVarColumns]
    
    dataframe_length = len(trainY)
    # dataframe_dim = Need to figure out how to count the columns of the array
    
    # reshape input to be [samples, time steps, features]
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    validateX = validateX.reshape(validateX.shape[0], 1, validateX.shape[1])
    
    
    ## The LSTM network expects the input data (X) to be provided with a specific
    # array structure in the form of: [samples, time steps, features]
    # Define the network
    modelFit = Sequential()
    modelFit.add(LSTM(20,
                      activation = 'sigmoid',            # sigmoid, relu, linear, softmax
                      input_shape = (1, 3)))
    modelFit.add(Dropout(.1))
    modelFit.add(Dense(1, activation = 'linear'))
    
    # Before training the model, configure the learning process via the compile method
    modelFit.compile(optimizer = 'adagrad',              # adam, adagrad
                     loss = 'mean_squared_error',        # poisson, mean_squared_error, binary_crossentropy
                     metrics = ['accuracy'])
    
    # Train the model
    modelEstimate = modelFit.fit(trainX, trainY,
                                 epochs = 100,
                                 batch_size = 1,
                                 verbose = 1,
                                 validation_data = (validateX, validateY))
    
    # make predictions
    trainPredict = modelFit.predict(trainX)
    validatePredict = modelFit.predict(validateX)
    
    # print the training accuracy and validation loss at each epoch
    # print the number of models of the network
    print(modelEstimate.history)
    print(len(modelFit.layers))
    
    
    # Invert predictions
    df_train = np.column_stack((trainPredict, train[:, 1:]))
    trainPredict2 = scaler.inverse_transform(df_train)
    
    df_validate = np.column_stack((validatePredict, validate[:, 1:]))
    validatePredict2 = scaler.inverse_transform(df_validate)
    
    
    # Plot the errors of the epochs and MSE
    plt.plot(modelEstimate.history['loss'])
    plt.plot(modelEstimate.history['val_loss'])
  #  plt.plot(modelEstimate.history['val_acc'])
    plt.title('Model Error History')
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Epochs')
    plt.legend(['Training Error', 'Validation Error'])
    plt.show()
    
    return;
    

    
# Read in the data
brentPriceDf = pd.read_csv('C:/GitRepos/EAA_Analytics/Personal/VT/Forecasts/Brent_DI_data.csv')
brentPriceDf.head(10)

# Remove the forecasts from the data
# inputData.BrentPrice[[inputData.Date > 2017-01-01 00:00:00]] = 0

# Format "Date" field as date
brentPriceDf[['Date']] = pd.to_datetime(brentPriceDf.Date)

# Convert the data frame to a Numpy array
brentPriceArr = brentPriceDf.iloc[:,0:].values


rnnModel(inputData = brentPriceArr, # read in the Boston housing data
         selectColumns = (1, 2, 3, 4),          # select columns from input dataset
         subsetXVarColumns = [1, 2, 3],              # select predictive features
         subsetYVarColumns = [0],                    # select target
         randSeed = 9                               # select random seed to reproduce results
         )
