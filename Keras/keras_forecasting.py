# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:31:25 2022

@author: Kasuni
"""

import numpy as np 
import pandas as pd

data = pd.read_csv('Datasets/DataSheet.csv', parse_dates=['month year'])
data.info()
data.isnull().sum()


data['month year'] = data['month year'].dt.to_period('M')
data['month year']=data['month year'].astype(str)


#Visualization
import matplotlib.pyplot as plt
plt.figure(figsize = (20, 10))


for i in range(len(data.columns)):
    if data.dtypes[i] != 'object':
        plt.subplot(3 ,5, i + 1)
        plt.boxplot(data[data.columns[i]] , vert = False)
        plt.title(data.columns[i])

#Preprocessing
def get_unique(df , columns):
    return {column : list(df[column].unique()) for column in columns}

categorical_columns = ['month year']

get_unique(data , categorical_columns)

ordinal_features = ['month year']


from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

import tensorflow as tf 

date_ordering = sorted(data['month year'].unique())
date_ordering

date_ordering.index('2022-01') # recent dates would be at the end, working as a timeline 

def ordinal_encode(df ,column , ordering):
    df = df.copy()
    df[column] = df[column].apply(lambda x : ordering.index(x))
    return df

def one_hot_encode(df , column):
    df = df.copy()
    dummies = pd.get_dummies(df[column])
    df = pd.concat([df , dummies] , axis = 1)
    df.drop(column , axis = 1 , inplace= True)
    return df

data = ordinal_encode(data , 'month year' , date_ordering)

target_column = 'Sales'

data.info()

data['OTIF'] = data['OTIF'].fillna(data['OTIF'].mean(), inplace=False)
data['OTIF'] = np.round(data['OTIF'], decimals = 2)

data['Embelishment Cost'] = data['Embelishment Cost'].fillna(data['Embelishment Cost'].mean(), inplace=False)
data['Embelishment Cost'] = np.round(data['Embelishment Cost'], decimals = 2)

#Splitting and Scalling
y = data[target_column]
X = data.drop(target_column , axis = 1)

scaler = StandardScaler() 

X = scaler.fit_transform(X)

X.shape , y.shape

X_train , X_test , y_train , y_test = train_test_split(X , y , train_size = 0.70)

#Training
inputs = tf.keras.Input(5,)
x = tf.keras.layers.Dense(64 , activation = 'relu')(inputs)
x = tf.keras.layers.Dense(64 , activation = 'relu')(x)
outputs = tf.keras.layers.Dense(1 , activation = 'sigmoid')(x)



model = tf.keras.Model(inputs = inputs , outputs = outputs)
model.compile(optimizer = 'adam' , 
             loss = 'mse' ,
             metrics = ['accuracy'])
model.summary()

history = model.fit(X_train , 
                    y_train , 
                    validation_split = 0.2 , 
                    batch_size = 64 , 
                    epochs = 100 , 
                    callbacks = [tf.keras.callbacks.ReduceLROnPlateau()]
                   )

#Result
batch_size = 64  
epochs = 100 

plt.figure(figsize = (14, 10))

epochs_range = range(1 , epochs + 1)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs_range , train_loss , label = 'Training loss')
plt.plot(epochs_range , val_loss , label = 'Validation loss')

plt.title('Training and Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

np.argmin(val_loss) + 1
model.evaluate(X_test , y_test)




userinputs = {'month year': ['2023-05'], 'Psc / Pk': [2], 'UnitPrice':[100], 'OTIF':[0.75], 'Embelishment Cost':2345} 
userinputs = pd.DataFrame(userinputs)  
scaler = StandardScaler() 
scaler1 = StandardScaler() 

userinputs = ordinal_encode(userinputs , 'month year' , '2023-05')

userinputs = scaler.fit_transform(userinputs)

scaled = model.predict(userinputs)
inversed = scaler.inverse_transform(userinputs)
print(inversed)