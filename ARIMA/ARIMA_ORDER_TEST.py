from statsmodels.tsa.stattools import adfuller
from numpy import log
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.arima_model import ARIMA   
# Import as Dataframe and converted to a series because specifying the index column.
df = pd.read_csv('./Datasets/C_datasets/C_dataset_monthly_original.csv', parse_dates=['Date'], index_col='Date')
df.head()



#Check for Staionary
result = adfuller(df.Price.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# Original Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.Price); axes[0, 0].set_title('Original Series')
plot_acf(df.Price, ax=axes[0, 1])

# 1st Differencing
data_difference_1 = df.Price.diff()
axes[1, 0].plot(df.Price.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.Price.diff().dropna(), ax=axes[1, 1])
result = adfuller(data_difference_1['Price'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# 2nd Differencing
axes[2, 0].plot(df.Price.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.Price.diff().diff().dropna(), ax=axes[2, 1])

plt.show()
y = df.Price

## Adf Test
ndiffs(y, test='adf')  # 2

# KPSS test
ndiffs(y, test='kpss')  # 0

# PP test:
ndiffs(y, test='pp')  # 2

#######
# Calculate the first difference of the time series
data_stationary = df.diff().dropna()
# Run ADF test on the differenced time series
result = adfuller(data_stationary['Price'])
# Plot the differenced time series
fig, ax = plt.subplots();
data_stationary.plot(ax=ax);
# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])

#Finding AR Terms (p)
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.Price.diff().dropna(), ax=axes[1])

plt.show()

#Finding MA Terms (q)
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1.2))
plot_acf(df.dropna(), ax=axes[1])

plt.show()

#How to bill the ARIMA model

# 1,1,2 ARIMA Model
model = ARIMA(df.Price, order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# 1,1,1 ARIMA Model
model = ARIMA(df.Price, order=(1,1,1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

########################################################################################
#How to do find the optimal ARIMA model manually using Out-of-Time Cross validation
from statsmodels.tsa.stattools import acf
# Create Training and Test
train = df.Price[:877]
test = df.Price[877:]

# Build Model
model = ARIMA(train, order=(1,1,0))  
#model = ARIMA(train, order=(1, 1, 1))  
fitted = model.fit(disp=-1) 
print(fitted.summary())

# Forecast
fc, se, conf = fitted.forecast(17, alpha=0.05)  # 95% conf


# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# multi-step out-of-sample forecast
start_index = len(test)
end_index = start_index + 6
forecast = fitted.predict(start=start_index, end=end_index)

# Forecast
fc1, se1, conf1 = fitted.forecast(10, alpha=0.05)  # 95% conf
# Make as pandas series
fc_series1 = pd.Series(fc1, index=test.index)






# Accuracy metrics
# Accuracy metrics
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
    acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy(fc, test.values)

#How to do Auto Arima Forecast in Python
import pmdarima as pm

#kf = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/wwwusage.csv', names=['value'], header=0)

model = pm.auto_arima(df.Price, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

