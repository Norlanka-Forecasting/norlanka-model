# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:08:05 2022

@author: Kasuni
"""

from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA   
import datetime
from statsmodels.tsa.stattools import acf
from sklearn.metrics import accuracy_score
import pickle

# Import as Dataframe and converted to a series because specifying the index column.
df = pd.read_csv('Datasets/DataSheet.csv',parse_dates=['month year'])
df.head()



df['month year'] = df['month year'].dt.to_period('M')
df = df.set_index('month year')
df.index= df.index.strftime('%Y-%m')
df['OTIF'] = df['OTIF'].fillna(df['OTIF'].mean(), inplace=False)
df['OTIF'] = np.round(df['OTIF'], decimals = 2)

df['Embelishment Cost'] = df['Embelishment Cost'].fillna(df['Embelishment Cost'].mean(), inplace=False)
df['Embelishment Cost'] = np.round(df['Embelishment Cost'], decimals = 2)

df['Sales'] = (df['Sales'] - df['Sales'].min()) / (df['Sales'].max() - df['Sales'].min())
df['Sales'] = np.round(df['Sales'], decimals = 2) 
df['Pcs / Pk'] = (df['Pcs / Pk'] - df['Pcs / Pk'].min()) / (df['Pcs / Pk'].max() - df['Pcs / Pk'].min()) 
df['Pcs / Pk'] = np.round(df['Pcs / Pk'], decimals = 2)
df['UnitPrice'] = (df['UnitPrice'] - df['UnitPrice'].min()) / (df['UnitPrice'].max() - df['UnitPrice'].min()) 
df['UnitPrice'] = np.round(df['UnitPrice'], decimals = 2)
df['OTIF'] = (df['OTIF'] - df['OTIF'].min()) / (df['OTIF'].max() - df['OTIF'].min()) 
df['OTIF'] = np.round(df['OTIF'], decimals = 2)
df['Embelishment Cost'] = (df['Embelishment Cost'] - df['Embelishment Cost'].min()) / (df['Embelishment Cost'].max() - df['Embelishment Cost'].min()) 
df['Embelishment Cost'] = np.round(df['Embelishment Cost'], decimals = 2)

import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = read_csv('Datasets/DataSheet.csv', usecols=[5], engine='python')
dataset = dataframe.values
print (dataset)
dataset = dataset.astype('float64')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]