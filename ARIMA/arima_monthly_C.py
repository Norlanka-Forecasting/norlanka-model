# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:08:05 2022

@author: Kasuni
"""

from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA   
import datetime
from statsmodels.tsa.stattools import acf
from sklearn.metrics import accuracy_score
import pickle

# Import as Dataframe and converted to a series because specifying the index column.
df_C = pd.read_csv('Datasets/C_datasets/C_dataset_monthly_original.csv', parse_dates=['Date'])
df_C.head()

df_C['Date'] = df_C['Date'].dt.to_period('M')
df_C = df_C.set_index('Date')
df_C.index= df_C.index.strftime('%Y-%m')
# Create Training and Test
train_C = df_C.Price[:29]
test_C = df_C.Price[28:]
df_C.plot()

# Build Model
model_C = ARIMA(train_C, order=(0,1,0))  
#model = ARIMA(train, order=(1, 1, 1))  
fitted_C = model_C.fit(disp=-1) 
print(fitted_C.summary())



min_month_C = "2022-05-01"
max_month_C = "2023-12-01"

months_C = pd.period_range(min_month_C, max_month_C, freq='M')
months_C = months_C.strftime('%Y-%m')


# Forecast
fc_C, se_C, conf_C = fitted_C.forecast(28, alpha=0.05)  # 95% conf

test_indexes_C = test_C.index
combined_C = test_indexes_C.union(months_C)
print(combined_C)
# Make as pandas series
fc_series_C = pd.Series(fc_C, index=combined_C)
lower_series_C = pd.Series(conf_C[:, 0], index=combined_C)
upper_series_C = pd.Series(conf_C[:, 1], index=combined_C)

# Plot
plt.figure(facecolor='#8C8181', dpi=150)
plt.plot(train_C, label='training')
plt.plot(test_C, label='actual')
plt.plot(fc_series_C, label='forecast')
plt.fill_between(lower_series_C.index, lower_series_C, upper_series_C, 
                 color='k', alpha=.15)
plt.xticks(rotation = 49, fontsize = 5)
#plt.title('Process Pineapple Price Market 2 Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('Month')
plt.ylabel('Price')
plt.title("Process Pineapple Market B Forecasting")
plt.show()



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
data = {
        "model": fitted_C, 
        "forecast_series": fc_series_C,
        }

with open('market_C_model.pkl', 'wb') as file:
    pickle.dump(data, file)
# from statsmodels.tsa.arima_model import ARIMAResults
# # load model
# loaded = ARIMAResults.load('model.pkl')
# import os;
# print(os.getcwd())
# 
# =============================================================================

print(len(df_C))
print(len(df_C)+30)


def getPrice_arima_123_C(Month):
    #pred123 = results.get_prediction(start=pd.to_datetime(Month), dynamic=False)
    pred123 = fc_series_C.get(key = Month)
    print(pred123)
    return pred123

result_C = fc_series_C.get(key = '2022-07')
getPrice_arima_123_C('2022-07')
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
#     acf1 = acf(fc_C-test_C)[1]                      # ACF1
#     return({'mape':mape, 'me':me, 'mae': mae, 
#             'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
#             'corr':corr, 'minmax':minmax})
# 
# forecast_accuracy(fc_C, test.values_C)
# 
# 
# acc_C=accuracy_score(test_C.values,fc_C)
# acc_C
# =============================================================================
