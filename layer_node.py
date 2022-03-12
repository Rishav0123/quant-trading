import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.6/bin")
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU,BatchNormalization, Dropout
from keras.activations import relu, sigmoid
from keras.layers import LeakyReLU

from sklearn.preprocessing import StandardScaler

#Encoding catagorical data
data1 = pd.read_csv('data_input.csv')
data2 = pd.read_csv('data_output.csv')
data1 = data1.fillna(0)
data2 = data2.fillna(0)
print(data1)
print(data2)

x = data1.iloc[:,1:].values
y = data2.iloc[:,1].values
train = pd.concat([data1.iloc[:,1:],data2.iloc[:,1:]], axis = 1)
print(train)
#Encoding catagorical data1
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x_1 = LabelEncoder()
#print(x[:, 13])
x[:, 13] = labelencoder_x_1.fit_transform(x[:, 13])

#splitting train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score, KFold
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim = x_train.shape[1]))
            model.add(Activation(activation))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
    model.add(Dense(1, kernel_initializer='normal', activation= 'linear'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

model = KerasRegressor(build_fn = create_model, verbose = 0)

print(model)

#layers =[[20], [40,20], [45,30,15], [128, 256, 256, 256]]
layers =[[45,30,15]]
activations = ['relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid_result = grid.fit(x_train, y_train)

print([grid_result.best_score_,grid_result.best_params_])
y_pred = grid.predict(x_test)

from sklearn.metrics import confusion_matrix
pred = pd.DataFrame(y_pred, columns = ['prediction'])
test = pd.DataFrame(y_test, columns = ['test'])
print(type(y_pred))
print(type(y_test))
pred_test = pd.concat([pred, test], axis=1)
print(pred_test)
print(pred_test.shape)
pred_test.to_csv("pred_test.csv")
