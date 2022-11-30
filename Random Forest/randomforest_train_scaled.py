
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

from sklearn.preprocessing import MinMaxScaler
# Get the referance for the data frame
# Import as Dataframe and converted to a series because specifying the index column.
df = pd.read_csv('Datasets/DataSheet.csv', parse_dates=['month year'])
df.head()

df = df.drop("month year", axis=1)

df['OTIF'] = df['OTIF'].fillna(df['OTIF'].mean(), inplace=False)
df['OTIF'] = np.round(df['OTIF'], decimals = 2)

df['Embelishment Cost'] = df['Embelishment Cost'].fillna(df['Embelishment Cost'].mean(), inplace=False)
df['Embelishment Cost'] = np.round(df['Embelishment Cost'], decimals = 2)

# =============================================================================
# df['Sales'] = (df['Sales'] - df['Sales'].min()) / (df['Sales'].max() - df['Sales'].min())
# df['Sales'] = np.round(df['Sales'], decimals = 2) 
# =============================================================================




# =============================================================================
# df['Pcs / Pk'] = (df['Pcs / Pk'] - df['Pcs / Pk'].min()) / (df['Pcs / Pk'].max() - df['Pcs / Pk'].min()) 
# df['Pcs / Pk'] = np.round(df['Pcs / Pk'], decimals = 2)
# df['UnitPrice'] = (df['UnitPrice'] - df['UnitPrice'].min()) / (df['UnitPrice'].max() - df['UnitPrice'].min()) 
# df['UnitPrice'] = np.round(df['UnitPrice'], decimals = 2)
# df['OTIF'] = (df['OTIF'] - df['OTIF'].min()) / (df['OTIF'].max() - df['OTIF'].min()) 
# df['OTIF'] = np.round(df['OTIF'], decimals = 2)
# df['Embelishment Cost'] = (df['Embelishment Cost'] - df['Embelishment Cost'].min()) / (df['Embelishment Cost'].max() - df['Embelishment Cost'].min()) 
# df['Embelishment Cost'] = np.round(df['Embelishment Cost'], decimals = 2)
# =============================================================================

# Train and Test Spiliting
X = df.drop("Sales", axis=1)
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

# Assigning the meadian values for other independent variables
Pcs_Pk = 103.0
UnitPrice = 1000.0
OTIF = 36989.5
Embelishment_Cost = 628.5

# Define the array with all input parameters
inputs = np.array([[Pcs_Pk, UnitPrice, OTIF,
               Embelishment_Cost]])
inputs = inputs.astype(float)

# Make the prediction
gross = random_forest_reg.predict(inputs)
result = "{:.2f}".format(gross[0])
print("Predicted Sales : ", round(gross[0]))

data = {
    "model": random_forest_reg,
    "train_features": train_features,
    "test_features": test_features,
    "train_labels": train_labels,
    "test_labels": test_labels,
    "y_pred": y_pred
}

# Expoert the model
with open('randomforest_model.pkl', 'wb') as file:
    pickle.dump(data, file)
