
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import pickle
import pandas as pd

# Get the referance for the data frame
# Import as Dataframe and converted to a series because specifying the index column.
df = pd.read_csv('Datasets/DataSheet.csv', parse_dates=['month year'])
df.head()



df['month year'] = df['month year'].dt.to_period('M')

df['month year']=df['month year'].astype(str)



#Preprocessing
def get_unique(df , columns):
    return {column : list(df[column].unique()) for column in columns}

categorical_columns = ['month year']

get_unique(df , categorical_columns)

ordinal_features = ['month year']
date_ordering = sorted(df['month year'].unique())
date_ordering

date_ordering.index('2022-01') # recent dates would be at the end, working as a timeline 

def ordinal_encode(df ,column , ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x : ordering.index(x))
    return df

df = ordinal_encode(df , 'month year' , date_ordering)


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

# Train and Test Spiliting
X = df.drop("Sales", axis=1)
X = X.drop("month year", axis=1)
y = df["Sales"]

# Split the datasetin to training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=1)

# Model
# RandomForestRegressor
random_forest_reg = RandomForestRegressor(n_estimators=1800, random_state=10)
random_forest_reg.fit(train_features, train_labels)
y_pred = random_forest_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("mean_squared_error : ${:,.02f}".format(error))
print("mean_absolute_percentage_error : ",
      mean_absolute_percentage_error(test_labels, y_pred))
print("max_error : ", max_error(test_labels, y_pred))
print("explained_variance_score : ",
      explained_variance_score(test_labels, y_pred))
print("mean_absolute_error : ", mean_absolute_error(test_labels, y_pred))
print("Score (R2): ", random_forest_reg.score(test_features, test_labels))
print("Score (R2): ", r2_score(test_labels, y_pred))



#######

# import datetime module
import datetime
 
# consider the start date as 2021-february 1 st
start_date = datetime.date(2017, 1, 1)
 
# consider the end date as 2021-march 1 st
end_date = datetime.date(2025, 3, 1)
 
# delta time
delta = datetime.timedelta(days=1)

forecastDates = []
i=0
 
# iterate over range of dates
while (start_date <= end_date):
    print(start_date, end="\n")
    start_date += delta
    forecastDates.insert(i, start_date)

forecastDates = pd.DataFrame (forecastDates, columns = ['column_name'])

forecastDates['column_name'] = pd.to_datetime(forecastDates['column_name'])
forecastDates['column_name'] = forecastDates['column_name'].dt.to_period('M')


forecastDates['column_name']=forecastDates['column_name'].astype(str)
date_ordering1 = sorted(forecastDates['column_name'].unique())
date_ordering1
forecastDates = ordinal_encode(forecastDates , 'column_name' , date_ordering1)
# Assigning the meadian values for other independent variables
month_year = '2023-02'
Pcs_Pk = 103.0
UnitPrice = 1000.0
OTIF = 36989.5
Embelishment_Cost = 628.5

# Define the array with all input parameters
inputs = np.array([[month_year, Pcs_Pk, UnitPrice, OTIF,
               Embelishment_Cost]])
inputs = pd.DataFrame(inputs, columns = ['Column_A','Column_B','Column_C','rt','yu'])
inputs = ordinal_encode(inputs , 'Column_A' , date_ordering1)

inputs = inputs.astype(float)

# Make the prediction
gross = random_forest_reg.predict(X)
result = "{:.2f}".format(gross[0])
print("Predicted Gross : $", result)