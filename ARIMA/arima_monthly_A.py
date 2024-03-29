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



# Create Training and Test
train = df.Sales[:292]
test = df.Sales[293:]
df.plot()



# Build Model
model = ARIMA(train, order=(0,1,0))
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit(disp=-1)
print(fitted.summary())
#
#
# min_month = "2022-05-01"
# max_month = "2023-12-01"
#
# months = pd.period_range(min_month, max_month, freq='M')
# months = months.strftime('%Y-%m')
#
# k=fitted.predict(start=1,end=2,dynamic=True)
# # Forecast
# fc, se, conf = fitted.forecast(28, alpha=0.05)  # 95% conf
#
# test_indexes = test.index
# combined = test_indexes.union(months)
# print(combined)
# # Make as pandas series
# fc_series = pd.Series(fc, index=combined)
# lower_series = pd.Series(conf[:, 0], index=combined)
# upper_series = pd.Series(conf[:, 1], index=combined)
#
# # Plot
# plt.figure( facecolor='#8C8181',dpi=200)
# plt.plot(train, label='training')
# plt.plot(test, label='actual')
# plt.plot(fc_series, label='forecast')
# plt.fill_between(lower_series.index, lower_series, upper_series,
#                  color='k', alpha=.15)
# plt.xticks(rotation = 49, fontsize = 5)
# #plt.title('Fresh Pineapple Price forecast vs Actuals')
# plt.legend(loc='upper left', fontsize=8)
# plt.xlabel('Month')
# plt.ylabel('Price')
# plt.title("Fresh Pineapple Market  Forecasting")
# plt.show()



# =============================================================================
# # multi-step out-of-sample forecast
# start_index = len(train)
# end_index =len(train)+len(test)-1
# #forecast = fitted.predict(start=start_index, end=end_index)
# pred=fitted.predict(start=start_index,end=end_index,typ='levels').rename('ARIMA predictions')
#  #pred.index=index_future_dates
# pred.plot(legend=True)
# test['Price'].plot(legend=True) 
# =============================================================================

# =============================================================================
# save model
# fitted.save('market_A_model.pkl')
# data = {
#         "model": fitted,
#         "forecast_series": fc_series,
#         }
#
# with open('market_A_model.pkl', 'wb') as file:
#     pickle.dump(data, file)
# from statsmodels.tsa.arima_model import ARIMAResults
# # load model
# loaded = ARIMAResults.load('model.pkl')
# import os;
# print(os.getcwd())
# 
# =============================================================================

# print(len(df))
# print(len(df)+30)
#
# def getPrice_arima_123_A(Month):
#     #pred123 = results.get_prediction(start=pd.to_datetime(Month), dynamic=False)
#     pred123 = fc_series.get(key = Month)
#     print(pred123)
#     return pred123
#
# result = fc_series.get(key = '2022-07')
# getPrice_arima_123_A('2022-07')
#pred=fitted.predict(start=1,end=1,typ='levels',dynamic=False).rename('ARIMA Predictions')

# =============================================================================
# def forecast_accuracy(forecast, actual):
#     mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
#     me = np.mean(forecast - actual)             # ME
#     mae = np.mean(np.abs(forecast - actual))    # MAE
#     mpe = np.mean((forecast - actual)/actual)   # MPE
#     rmse = np.mean((forecast - actual)**2)**.5  # RMSE
#     corr = np.corrcoef(forecast, actual)[0,1]   # corr
#     mins = np.amin(np.hstack([forecast[:,None], 
#                               actual[:,None]]), axis=1)
#     maxs = np.amax(np.hstack([forecast[:,None], 
#                               actual[:,None]]), axis=1)
#     minmax = 1 - np.mean(mins/maxs)             # minmax
#     acf1 = acf(fc-test)[1]                      # ACF1
#     return({'mape':mape, 'me':me, 'mae': mae, 
#             'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
#             'corr':corr, 'minmax':minmax})
# 
# forecast_accuracy(fc, test.values)
# 
# 
# acc=accuracy_score(test.values,fc)
# acc
# =============================================================================
