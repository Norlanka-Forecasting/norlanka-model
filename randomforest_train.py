
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import data_preprocessing
import pickle

# Get the referance for the data frame
df = data_preprocessing.df
country_le = data_preprocessing.country

# Train and Test Spiliting
X = df.drop("gross", axis=1)
y = df["gross"]

# Split the datasetin to training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=1)

# Model
# RandomForestRegressor
random_forest_reg = RandomForestRegressor(n_estimators=1800, random_state=10)
random_forest_reg.fit(train_features, train_labels)
y_pred = random_forest_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("root_mean_squared_error : ${:,.02f}".format(error))
print("explained_variance_score : ",
      explained_variance_score(test_labels, y_pred))
print("Score (R2): ", r2_score(test_labels, y_pred))

data = {
    "model": random_forest_reg,
    "country_le": country_le,
    "train_features": train_features,
    "test_features": test_features,
    "train_labels": train_labels,
    "test_labels": test_labels,
    "y_pred": y_pred
}

# Export the model
with open('randomforest_model.pkl', 'wb') as file:
    pickle.dump(data, file)
