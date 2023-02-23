import numpy as np
import time

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K, optimizers
import h5py
import os
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 全局变量
batch_size = 32
nb_classes = 2
epochs = 1
name = 'epsilon'

model = Sequential()
model.add(Dense(1024))  # 全连接层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层

model.add(Dense(256))  # 全连接层2
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层

model.add(Dense(2))  # 全连接层3
model.add(Activation('softmax'))  # Softmax评分

model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

alls = 0
for num in range(10):
    X_train = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__x_train.npy')
    y_train = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__y_train.npy')
    # 转换为one_hot类型
    Y_train = np_utils.to_categorical(y_train, nb_classes)

    starttime = datetime.datetime.now()

    # 训练模型
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
    # model.fit(X_train1, Y_train1, batch_size=batch_size, epochs=1, verbose=1, validation_data=(X_test, Y_test))

    # long running
    endtime = datetime.datetime.now()
    alls = alls + (endtime - starttime).seconds
    print((endtime - starttime).seconds)

print(alls)

# alls = 0
# for num in range(10):
#     X_test = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__x_test.npy')
#     y_test = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__y_test.npy')
#     # 转换为one_hot类型
#     Y_test = np_utils.to_categorical(y_test, nb_classes)
#     print(model.evaluate(X_test, Y_test))
#     alls = alls + model.evaluate(X_test, Y_test)[1]
#
# print(alls/10)