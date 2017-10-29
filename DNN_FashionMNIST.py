#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:26:24 2017

@author: macro
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from keras.optimizers import SGD
from sklearn.model_selection import cross_val_score
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

seed = 50
np.random.seed(seed)


# load dataset
dataframe = pd.read_csv("./fashion-mnist_train.csv")
dataset_train = dataframe.values

dataframe = pd.read_csv("./fashion-mnist_test.csv")
dataset_test = dataframe.values

X_train = np.array(dataset_train[:,1:].astype('float32'))
X_train = X_train / 255

y_train = np.array(dataset_train[:,0])
# 멀티 클래스로 분류되니까, 원 핫 인코딩 한다.
y_train = np_utils.to_categorical(y_train)


X_test = np.array(dataset_test[:,1:].astype('float32'))
X_test = X_test / 255

y_test = np.array(dataset_test[:,0])
# 멀티 클래스로 분류되니까, 원 핫 인코딩 한다.
y_test = np_utils.to_categorical(y_test)

# 변수 셋팅
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# 모델 만들기
def create_model():

    model = Sequential()
    model.add(Dense(input_dim, input_dim=input_dim, kernel_initializer='normal', activation='relu',
    kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(400, kernel_initializer='normal', activation='relu',
    kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(100, kernel_initializer='normal', activation='relu',
    kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim, kernel_initializer='normal', activation='softmax'))
    
    # Compile model
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = create_model()

# fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)

acc = scores[1] * 100
print("Accuracy: %.2f%% " % acc )


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()





