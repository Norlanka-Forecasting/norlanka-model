import warnings
import itertools
import pandas as pd
import numpy as np
from math import sqrt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller 
from sklearn.metrics import mean_squared_error 
plt.style.use('fivethirtyeight')


data = pd.read_csv('Datasets/A_datasets/A_dataset_daily_original.csv',parse_dates=True,index_col='Date')
print(data)
y = data



y.plot(figsize=(15, 6))
plt.show()

def test_stationarity(timeseries,maxlag):
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries,maxlag=maxlag,
                      autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (round(dfoutput,3))
test_stationarity(y,1)



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
     for param_seasonal in seasonal_pdq:
         try:
             mod = sm.tsa.statespace.SARIMAX(y,
                                             order=param,
                                             seasonal_order=param_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)

             results = mod.fit()

             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
         except:
             continue
         
train = y.Price[:877]
test = y.Price[877:]
        
mod = sm.tsa.statespace.SARIMAX(train,
                                order=(1, 0, 1),
                                seasonal_order=(0 ,1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

pred = results.get_prediction(start=pd.to_datetime('2021-09-24'), end=pd.to_datetime('2022-04-30'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['2021':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('Prices')
plt.legend()

plt.show()

y_forecasted = pred.predicted_mean
#y_truth = y['2019-05-01':]
y_truth = test

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# report performance
rmse = sqrt(mean_squared_error(y_truth, y_forecasted))
print(rmse)

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=2045)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()



def getPrice_arima(Month):
    print(Month)
    print(pd.to_datetime(Month))
    pred123 = results.get_prediction(start=pd.to_datetime(Month), dynamic=False)
    pred123 = pred123.predicted_mean
    return pred123.values[0]

getPrice_arima('2022-07-01')

