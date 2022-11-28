from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
import data_preprocessing
from sklearn.linear_model import LinearRegression

# defining the preprocessing data
df = data_preprocessing.df
country_le = data_preprocessing.country

## Defining independent and dependent variables
X = df.drop("gross", axis=1)
y = df["gross"]

## Split Data into train and testing datasets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size = 0.25, random_state = 1)

## Model

#LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(train_features, train_labels)
y_pred = linear_reg.predict(test_features)

## Evaluate the model's performance
print(" ")
print("Accuracy measures of Linear Regression")
print(" ")
print("explained_variance_score : ",explained_variance_score(test_labels, y_pred))
print("Score (R2): ",r2_score(test_labels, y_pred))

# Calculate mean squared value  
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("Root mean_squared_error : ${:,.02f}".format(error))

# print("mean_absolute_percentage_error : ",mean_absolute_percentage_error(test_labels, y_pred))
# print("max_error : ",max_error(test_labels, y_pred))
# print("explained_variance_score : ",explained_variance_score(test_labels, y_pred))
# print("mean_absolute_error : ", mean_absolute_error(test_labels, y_pred))
# print("Score (R2): ", linear_reg.score(test_features, test_labels))
# print("Score (R2): ",r2_score(test_labels, y_pred))








