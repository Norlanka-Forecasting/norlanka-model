# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
# load dataset

# Import as Dataframe and converted to a series because specifying the index column.
series = pd.read_csv('Datasets/DataSheet.csv',parse_dates=['month year'])
series.head()


series = series.sort_values(by='month year',ascending=True)
series['month year'] = series['month year'].dt.to_period('M')
series = series.set_index('month year')
series.index= series.index.strftime('%Y-%m')


series = series.drop("OTIF", axis=1)
series = series.drop("Pcs / Pk", axis=1)
series = series.drop("UnitPrice", axis=1)
series = series.drop("Embelishment Cost", axis=1)
# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()