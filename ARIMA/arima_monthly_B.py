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
df_B = pd.read_csv('Datasets/B_datasets/B_dataset_monthly_original.csv', parse_dates=['Date'])
df_B.head()

df_B['Date'] = df_B['Date'].dt.to_period('M')
df_B = df_B.set_index('Date')
df_B.index= df_B.index.strftime('%Y-%m')
# Create Training and Test
train_B = df_B.Price[:29]
test_B = df_B.Price[28:]
df_B.plot()

# Build Model
model_B = ARIMA(train_B, order=(0,1,0))  
#model = ARIMA(train, order=(1, 1, 1))  
fitted_B = model_B.fit(disp=-1) 
print(fitted_B.summary())



min_month = "2022-05-01"
max_month = "2023-12-01"

months_B = pd.period_range(min_month, max_month, freq='M')
months_B = months_B.strftime('%Y-%m')


# Forecast
fc_B, se_B, conf_B = fitted_B.forecast(28, alpha=0.05)  # 95% conf

test_indexes_B = test_B.index
combined_B = test_indexes_B.union(months_B)
print(combined_B)
# Make as pandas series
fc_series_B = pd.Series(fc_B, index=combined_B)
lower_series_B = pd.Series(conf_B[:, 0], index=combined_B)
upper_series_B = pd.Series(conf_B[:, 1], index=combined_B)

# Plot
plt.figure(facecolor='#8C8181', dpi=200)
plt.plot(train_B, label='training')
plt.plot(test_B, label='actual')
plt.plot(fc_series_B, label='forecast')
plt.fill_between(lower_series_B.index, lower_series_B, upper_series_B, 
                 color='k', alpha=.15)
plt.xticks(rotation = 40, fontsize = 5)
#plt.title('Process Pineapple Price Market 1 Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.xlabel('Month')
plt.ylabel('Price')
plt.title("Process Pineapple Market A Forecasting")
plt.show()


# =============================================================================
# save model
data = {
        "model": fitted_B, 
        "forecast_series": fc_series_B,
        }

with open('market_B_model.pkl', 'wb') as file:
    pickle.dump(data, file)
# from statsmodels.tsa.arima_model import ARIMAResults
# # load model
# loaded = ARIMAResults.load('model.pkl')
# import os;
# print(os.getcwd())
# 
# =============================================================================
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

print(len(df_B))
print(len(df_B)+30)


def getPrice_arima_123_B(Month):
    #pred123 = results.get_prediction(start=pd.to_datetime(Month), dynamic=False)
    pred123 = fc_series_B.get(key = Month)
    print(pred123)
    return pred123

result = fc_series_B.get(key = '2022-07')
getPrice_arima_123_B('2022-07')
#pred=fitted.predict(start=1,end=1,typ='levels',dynamic=False).rename('ARIMA Predictions')
# =============================================================================
# 
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
#     acf1 = acf(fc_B-test_B)[1]                      # ACF1
#     return({'mape':mape, 'me':me, 'mae': mae, 
#             'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
#             'corr':corr, 'minmax':minmax})
# 
# forecast_accuracy(fc_B, test_B.values)
# 
# 
# acc_B=accuracy_score(test_B.values,fc_B)
# acc_B
# =============================================================================
