import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
df = pd.read_csv("movie_metadata.csv")

df.head()

df = df[["country", "budget", "imdb_score", "gross"]]

df.head()

# remove raws with null values
df = df.dropna()
df.isnull().sum()
df.head()


#df_usa = df[df['country'] == 'USA']
#df_usa_new = df_usa.iloc[:400]


#df = df[df.country != 'USA']

#df = [df, df_usa_new]
#df = pd.concat(df)


# split the genres types an assing a rating
#df_tepm = df['genres'].str.split('|', expand=True)
#df['genres_def'] = df_tepm.count(axis=1)

# remove the genres colomn
#df = df.drop('genres', axis=1)

# len(df['country'].unique())

def cutoff_countries(countries, limit):
    country_map = {}
    for i in range(len(countries)):
        if countries.values[i] < limit:
            country_map[countries.index[i]] = 'Other'
        else:
            country_map[countries.index[i]] = countries.index[i]

    return country_map


country_map = cutoff_countries(df.country.value_counts(), 80)
df['country'] = df['country'].map(country_map)
df.country.value_counts()

df['budget'].max()
df['budget'].min()
df = df[df['country'] != 'Other']

df['imdb_score'] = df['imdb_score'].round(decimals=0)

country = LabelEncoder()
df['country'] = country.fit_transform(df['country'])
df["country"].unique()

#language = LabelEncoder()
#df['language'] = language.fit_transform(df['language'])
# df["language"].unique()

# Min-Max Normalization
#df = data.drop('species', axis=1)
df_ori = df
df = (df-df.min())/(df.max()-df.min())


X = df.drop("gross", axis=1)
y = df["gross"]

train_features, test_features, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25, random_state=1)

# LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(train_features, train_labels)
y_pred = linear_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))
print("mean_absolute_percentage_error : ",
      mean_absolute_percentage_error(test_labels, y_pred))
print("max_error : ", max_error(test_labels, y_pred))
print("explained_variance_score : ",
      explained_variance_score(test_labels, y_pred))
print("mean_absolute_error : ", mean_absolute_error(test_labels, y_pred))


plt.figure(figsize=(10, 10))
plt.scatter(test_labels, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(y_pred), max(test_labels))
p2 = min(min(y_pred), min(test_labels))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()

linear_reg.summary()

test_score = linear_reg.score(test_labels, y_pred)
print("The score of the model on test data is:", test_score)

# DecisionTreeRegressor
dec_tree_reg = DecisionTreeRegressor(random_state=0)
dec_tree_reg.fit(train_features, train_labels)
y_pred = dec_tree_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))
print("mean_absolute_percentage_error : ",
      mean_absolute_percentage_error(test_labels, y_pred))
print("max_error : ", max_error(test_labels, y_pred))
print("explained_variance_score : ",
      explained_variance_score(test_labels, y_pred))
print("mean_absolute_error : ", mean_absolute_error(test_labels, y_pred))


p1 = max(max(y_pred), max(test_labels))
p2 = min(min(y_pred), min(test_labels))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()
# =============================================================================
# test_score = dec_tree_reg.score(test_labels, y_pred)
# print("The score of the model on test data is:", test_score )
# =============================================================================

# RandomForestRegressor
random_forest_reg = RandomForestRegressor(random_state=0)
random_forest_reg.fit(train_features, train_labels)
y_pred = random_forest_reg.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))
print("mean_absolute_percentage_error : ",
      mean_absolute_percentage_error(test_labels, y_pred))
print("max_error : ", max_error(test_labels, y_pred))
print("explained_variance_score : ",
      explained_variance_score(test_labels, y_pred))
print("mean_absolute_error : ", mean_absolute_error(test_labels, y_pred))


max_depth = [None, 2, 4, 6, 8, 10, 12]
parameters = {"max_depth": max_depth}

regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
gs.fit(X, y.values)

regressor = gs.best_estimator_

regressor.fit(X, y.values)
y_pred = regressor.predict(X)
error = np.sqrt(mean_squared_error(y, y_pred))
print("${:,.02f}".format(error))

classifier = LogisticRegression(random_state=0)
classifier.fit(train_features, train_labels)
y_pred = classifier.predict(test_features)
error = np.sqrt(mean_squared_error(test_labels, y_pred))
print("${:,.02f}".format(error))

accuracy = accuracy_score(test_labels, y_pred)

X = np.array([["USA", 2.37e+08, 7]])
X

X[:, 0] = country.transform(X[:, 0])
#X[:, 2] = language.transform(X[:,2])
X = X.astype(float)
X

y_pred = random_forest_reg.predict(X)
y_pred = (y_pred * (df_ori.max() - df_ori.min()) + df_ori.min())
y_pred
df.head()
