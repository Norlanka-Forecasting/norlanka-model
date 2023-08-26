# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:28:45 2022

@author: Kasuni
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df = pd.read_csv("Datasets/DataSheet.csv")

### Find out most Important Features ####
# Plotting the Correlation between the numerical values of the Dataset
correlations = df.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)

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

## How would the model look like in 3D space with 2 most important features.

#Get train features and labels for training dataset.
X_users_reviews_train = train_features[['Pcs / Pk', 'UnitPrice']].values.reshape(-1,2)
Y_gross_train_2 = train_labels.values

#Get test features and labels for testing dataset.
X_users_reviews__test = test_features[['Pcs / Pk', 'UnitPrice']].values.reshape(-1,2)
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
    ax.set_xlabel('Pcs / Pk', fontsize=12)
    ax.set_ylabel('UnitPrice', fontsize=12)
    ax.set_zlabel('Sales', fontsize=12)
    ax.locator_params(nbins=4, axis='x')
    ax.locator_params(nbins=5, axis='x')
########
ax1.text2D(0.2, 0.32, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax1.transAxes, color='grey', alpha=0.5)
ax2.text2D(0.3, 0.42, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax2.transAxes, color='grey', alpha=0.5)
ax3.text2D(0.85, 0.85, 'aegis4048.github.io', fontsize=13, ha='center', va='center',
           transform=ax3.transAxes, color='grey', alpha=0.5)

ax1.view_init(elev=28, azim=120)
ax2.view_init(elev=4, azim=114)
ax3.view_init(elev=60, azim=165)

fig.suptitle('$R^2 = %.2f$' % r2, fontsize=20)

fig.tight_layout()
