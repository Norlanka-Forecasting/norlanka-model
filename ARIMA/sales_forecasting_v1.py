
from pandas import read_csv
from matplotlib import pyplot
series = read_csv('Datasets/DataSheet.csv', header=0, index_col=0, parse_dates=['month year'])


series = series.drop("Embelishment Cost", axis=1)
series = series.drop("OTIF", axis=1)
series = series.drop("UnitPrice", axis=1)
series = series.drop("Pcs / Pk", axis=1)

series.plot()
pyplot.show()

resample = series.resample('M')
monthly_mean = resample.mean()
print(monthly_mean.head(13))
monthly_mean.plot()
pyplot.show()

