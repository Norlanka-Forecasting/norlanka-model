# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:27:08 2022

@author: Kasuni
"""

import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from pmdarima.arima.utils import ndiffs
import numpy as np
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.api import SimpleExpSmoothing

data =  pd.read_csv('Datasets/C_datasets/C_dataset_daily_original.csv', parse_dates=['Date'], index_col=['Date'])

#today = datetime.date.today()
plt.figure()
plt.title('Fresh Pineapple Price Forecasting')
plt.plot(data['Price'])

SimpleExpSmoothing(data).fit(smoothing_level=0.1)

data = data['Price'].tolist()

train_data = data[0:877]
test_data = data[877:1096]

all_index = pd.date_range(start=pd.to_datetime('2019-05-01'), end=pd.to_datetime('2022-04-30'), freq='D')
all_price_data =  pd.Series(data, all_index)

train_index= pd.date_range(start=pd.to_datetime('2019-05-01'), end=pd.to_datetime('2021-09-23'), freq='D')
train_price_data = pd.Series(train_data, train_index)

test_index= pd.date_range(start=pd.to_datetime('2021-09-24'), end=pd.to_datetime('2022-04-30'), freq='D')
test_price_data = pd.Series(test_data, test_index)

forecast_timestep = 219

fit_1 = SimpleExpSmoothing(train_price_data, initialization_method="heuristic").fit(smoothing_level=0.1,optimized=False)
forecast1 = fit_1.forecast(forecast_timestep).rename(r'$\alpha=0.1$')
plt.plot(all_price_data, color='black') #actual values
plt.plot(fit_1.fittedvalues, color='cyan') #trained  values
line1, = plt.plot(forecast1, color='red') #forecasted values/tested values


fit_2 = SimpleExpSmoothing(train_price_data, initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
forecast2 = fit_2.forecast(forecast_timestep).rename(r'$\alpha=0.4$')
plt.plot(all_price_data, color='black') #actual values
plt.plot(fit_1.fittedvalues, color='cyan') #trained  values
line1, = plt.plot(forecast1, color='red') #forecasted values/tested values


fit_3 = SimpleExpSmoothing(train_price_data, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
forecast3 = fit_3.forecast(forecast_timestep).rename(r'$\alpha=0.8$')
plt.plot(all_price_data, color='black') #actual values
plt.plot(fit_1.fittedvalues, color='cyan') #trained  values
line1, = plt.plot(forecast1, color='red') #forecasted values/tested values


fit_4 = SimpleExpSmoothing(train_price_data, initialization_method="estimated").fit()
forecast4 = fit_4.forecast(forecast_timestep).rename(r'$\alpha=%s$'%fit_4.model.params['smoothing_level'])
plt.plot(all_price_data, color='black') #actual values
plt.plot(fit_1.fittedvalues, color='cyan') #trained  values
line1, = plt.plot(forecast1, color='red') #forecasted values/tested values



def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(forecast-test_price_data)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(forecast1.values, test_price_data.values)
forecast_accuracy(forecast2.values, test_price_data.values)
forecast_accuracy(forecast3.values, test_price_data.values)
forecast_accuracy(forecast4.values, test_price_data.values)



