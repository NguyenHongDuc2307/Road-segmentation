import pandas as pd
import numpy as np
from datetime import datetime
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer, Dense 
from PIL import Image
#print(keras.__version__)
# checking version of packages
#print('Version of sklearn: ', sklearn.__version__)
#print('Version of tensorflow: ', tf.__version__)


DataFileName = 'airs-dataset/data.csv'
data = pd.read_csv(DataFileName)
#print(valid.head())

X = data.drop('76', axis=1)
y = data['76']
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=data['76'],random_state=10,test_size=0.3)

input_neurons = X_train.shape[1]
output_neurons = 1
number_of_hidden_layers = 3
neuron_hidden_layer_1 = 100
neuron_hidden_layer_2 = 50
neuron_hidden_layer_3 = 30

model = Sequential()
model.add(InputLayer(input_shape=(input_neurons,)))
model.add(Dense(units=neuron_hidden_layer_1, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_2, activation='relu'))
model.add(Dense(units=neuron_hidden_layer_3, activation='relu'))
model.add(Dense(units=output_neurons, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
K.set_value(model.optimizer.learning_rate, 0.001)
model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5)
model.save('model')
prediction = model.predict_classes(X_test)
print(accuracy_score(y_test, prediction))

