#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:09:51 2017

@author: macro
"""

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
# fix dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')


seed = 50
np.random.seed(seed)

# load dataset
dataframe = pd.read_csv("./fashion-mnist_train.csv")
dataset_train = dataframe.values

dataframe = pd.read_csv("./fashion-mnist_test.csv")
dataset_test = dataframe.values

X_train = np.array(dataset_train[:,1:].astype('float32'))
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_train = X_train / 255.0

y_train = np.array(dataset_train[:,0])
# 멀티 클래스로 분류되니까, 원 핫 인코딩 한다.
y_train = np_utils.to_categorical(y_train)


X_test = np.array(dataset_test[:,1:].astype('float32'))
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_test = X_test / 255.0

y_test = np.array(dataset_test[:,0])
# 멀티 클래스로 분류되니까, 원 핫 인코딩 한다.
y_test = np_utils.to_categorical(y_test)


# 분류할 갯수
output_dim = y_train.shape[1]

# CNN 모델 만들기

def create_model():
    model = Sequential()
    model.add(Conv2D(20, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(40, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(80, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = create_model()

# fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=56)
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



