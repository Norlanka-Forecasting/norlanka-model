# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 01:06:04 2022

@author: isira
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

df = df.sort_values(by='month year',ascending=True)
df['month year'] = df['month year'].dt.to_period('M')

#df = df.set_index('month year')
#df.index= df.index.strftime('%Y-%m')
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

data = df.drop(['OTIF'], axis=1)
data = data.drop(['month year'], axis=1)
data.index = df['month year']



train = data[:int(0.8*(len(data)))]
valid = data[int(0.8*(len(data))):]

#fit the model
from statsmodels.tsa.vector_ar.var_model import VAR

model = VAR(endog=train)
model_fit = model.fit()

# make prediction on validation
prediction = model_fit.forecast(model_fit.y, steps=len(valid))

cols = data.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,4):
    for i in range(0, len(prediction)):
       pred.iloc[i][j-1] = prediction[i][j]


from math import sqrt
from sklearn.metrics import mean_squared_error

#check rmse
for i in data.columns:
    print('rmse value for', i, 'is : ', sqrt(mean_squared_error(pred[i], valid[i])))
