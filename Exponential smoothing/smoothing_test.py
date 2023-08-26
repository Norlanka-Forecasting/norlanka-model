# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:35:01 2022

@author: Kasuni
"""

import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

data = yf.download(tickers='MSFT', period='1mo', interval='1d')
data =  pd.read_csv('Datasets/A_datasets/A_dataset_monthly_original.csv', parse_dates=['Date'], index_col=['Date'])

#today = datetime.date.today()
plt.figure()
plt.title('Fresh Pineapple Price Forecasting')
plt.plot(data['Price'])

SimpleExpSmoothing(data).fit(smoothing_level=0.1)

data = data['Price'].tolist()

index= pd.date_range(start=pd.to_datetime('2019-05-01'), end=pd.to_datetime('2022-04-30'), freq='M')
price_data = pd.Series(data, index)
forecast_timestep = 2

fit_1 = SimpleExpSmoothing(price_data, initialization_method="heuristic").fit(smoothing_level=0.1,optimized=False)
forecast1 = fit_1.forecast(forecast_timestep).rename(r'$\alpha=0.1$')
plt.plot(price_data, color='black') #actual values
plt.plot(fit_1.fittedvalues, color='cyan') #predicted values
line1, = plt.plot(forecast1, color='cyan')

fit_2 = SimpleExpSmoothing(price_data, initialization_method="heuristic").fit(smoothing_level=0.4,optimized=False)
forecast2 = fit_2.forecast(forecast_timestep).rename(r'$\alpha=0.4$')
plt.plot(fit_2.fittedvalues, color='red') #train data
line2, = plt.plot(forecast2, color='green') #predicted values

fit_3 = SimpleExpSmoothing(price_data, initialization_method="heuristic").fit(smoothing_level=0.6,optimized=False)
forecast3 = fit_3.forecast(forecast_timestep).rename(r'$\alpha=0.8$')

fit_4 = SimpleExpSmoothing(price_data, initialization_method="estimated").fit()
forecast4 = fit_4.forecast(forecast_timestep).rename(r'$\alpha=%s$'%fit_4.model.params['smoothing_level'])

plt.figure(figsize=(16,10))
plt.plot(price_data,  color='black') #actual values
plt.plot(fit_1.fittedvalues,  color='cyan') #fit_1 train data
line1, = plt.plot(forecast1,  color='cyan') #fit_1 - forcasted data
plt.plot(fit_2.fittedvalues,  color='red') #fit_2 train data
line2, = plt.plot(forecast2, color='red') #fit_2 forecasted data
plt.plot(fit_3.fittedvalues,  color='green') #fit_3 train data
line3, = plt.plot(forecast3, color='green')#fit_3 forecasted data
plt.plot(fit_4.fittedvalues,  color='blue') #fit_4 train data
line4, = plt.plot(forecast4, color='blue') #fit_4 forecasted data
plt.legend([line1, line2, line3,line4], [forecast1.name, forecast2.name, forecast3.name,forecast4.name])
plt.show()