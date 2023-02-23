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
from line_profiler import LineProfiler
lp = LineProfiler()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 全局变量
batch_size = 32
nb_classes = 2
epochs = 1


model = Sequential()
model.add(Dense(32))  # 全连接层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层

model.add(Dense(1))  # 全连接层3

model.compile(loss='mse',
              optimizer="sgd",
              metrics=['mse'])


X_train = np.load('/data/USA_Housing/USA_Housing_x_train_tf.npy')
Y_train = np.load('/data/USA_Housing/USA_Housing_y_train_tf.npy')

print(Y_train.shape)


# starttime = datetime.datetime.now()

# 训练模型
def hanshu():
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=10)
# model.fit(X_train1, Y_train1, batch_size=batch_size, epochs=1, verbose=1, validation_data=(X_test, Y_test))

lp_wrapper = lp(hanshu)
lp_wrapper()
lp.print_stats()

# # long running
# endtime = datetime.datetime.now()
# print((endtime - starttime).seconds)

X_test = np.load('/data/USA_Housing/USA_Housing_x_test_tf.npy')
Y_test = np.load('/data/USA_Housing/USA_Housing_y_test_tf.npy')

print(model.evaluate(X_test, Y_test))

