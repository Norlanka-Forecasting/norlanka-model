#Import libraries required for prediction and visualization.
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

#Function to load trained model.
def load_model():
    with open('randomforest_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

#Load trained model.
data = load_model()

#Get required variables from loaded model.
random_forest_reg = data["model"]

test_features = data["test_features"]
test_labels = data["test_labels"]
train_features = data["train_features"]
train_labels = data["train_labels"]
y_pred = data["y_pred"]

##  Feature importances are obtained through rfpimp library.
import rfpimp
#Obtain importances of testing dataset.
imp = rfpimp.importances(random_forest_reg, test_features, test_labels)

#Plot multiple plots in single figure.
fig, ax = plt.subplots(figsize=(6, 3))

#Make the horizontal bar plot.
ax.barh(imp.index, imp['Importance'], height=0.8, facecolor='grey', alpha=0.8, edgecolor='k')
#Set the label for the x-axis.
ax.set_xlabel('Importance score')
#Set the title for the Axes.
ax.set_title('Permutation feature importance')

#Get most important feature to top.
plt.gca().invert_yaxis()
fig.tight_layout()

##Most Important Feature with Predicted Feature (Actual Tested and Precited) After Spilliting.

#Get test features and labels for testing dataset.
X_users_test = test_features['num_voted_users'].values.reshape(-1,1)
y_gross = test_labels.values

#Get train features and labels for training dataset.
X_users_train = train_features['num_voted_users'].values.reshape(-1,1)
y_gross_train = train_labels.values

#Train model and get prediction with most important feature.
ols = RandomForestRegressor(random_state=0)
model = ols.fit(X_users_train, y_gross_train)
response = model.predict(X_users_test)
#Get r2 score for trained model.
r2 = model.score(X_users_test, y_gross)

plt.style.use('default')
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(8, 4))

#Plot test values with prediction values.
ax.plot(X_users_test, y_pred, color='black', label='Random Forest Regression model')
ax.scatter(X_users_test, y_gross, edgecolor='k', facecolor='grey', alpha=0.7, label='Sample data')
#Set the labels for y-axis and x-axis.
ax.set_ylabel('Gross', fontsize=14)
ax.set_xlabel('Number of voted Users', fontsize=14)

#Place the legend on the Axes.
ax.legend(facecolor='white', fontsize=11)
#Set the title for the Axes to r2 score value.
ax.set_title('$R^2= %.2f$' % r2, fontsize=18)

fig.tight_layout()

## How would the model look like in 3D space with 2 most important features.

#Get train features and labels for training dataset.
X_users_reviews_train = train_features[['num_voted_users', 'budget']].values.reshape(-1,2)
Y_gross_train_2 = train_labels.values

#Get test features and labels for testing dataset.
X_users_reviews__test = test_features[['num_voted_users', 'budget']].values.reshape(-1,2)
y_gross_test_2 = test_labels.values

#Set x, y, z axes.
x = X_users_reviews_train[:, 0]
y = X_users_reviews_train[:, 1]
z = Y_gross_train_2

x_pred = np.linspace(6, 24, 30)   # range of num_voted_users values
y_pred = np.linspace(0, 100, 30)  # range of budget values

xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

#Train model with selected features.
ols = RandomForestRegressor(n_estimators = 1000, random_state = 15)
model = ols.fit(X_users_reviews_train, Y_gross_train_2)
predicted = model.predict(model_viz)
#Get r2 score to evaluate model.
r2 = model.score(X_users_reviews__test, y_gross_test_2)

############################################## Plot ################################################

plt.style.use('default')

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

axes = [ax1, ax2, ax3]

for ax in axes:
    ax.plot(x, y, z, color='k', zorder=15, linestyle='none', marker='o', alpha=0.5)
    ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
    ax.set_xlabel('num_voted_users', fontsize=12)
    ax.set_ylabel('budget', fontsize=12)
    ax.set_zlabel('Gross', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()

##Plot to show the spread of the true values and predicted values of the testing dataset.

test_features = data["test_features"]
test_labels = data["test_labels"]
train_features = data["train_features"]
train_labels = data["train_labels"]
y_pred = data["y_pred"]

plt.figure(figsize=(10, 10))
#Plot test values with prediction of initially loaded moel.
plt.scatter(test_labels, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')

#Set maximum and minimum.
p1 = max(max(y_pred), max(test_labels))
p2 = min(min(y_pred), min(test_labels))
#Plot line graph based on maximum and minimum.
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.show()