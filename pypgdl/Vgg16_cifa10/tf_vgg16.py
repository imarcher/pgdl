import numpy as np
import time
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K, optimizers
import h5py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# 全局变量
batch_size = 128
nb_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 32, 32
# number of convolutional filters to use
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(np.shape(X_train), np.shape(y_train))
print(np.shape(X_test), np.shape(y_test))
print(type(X_train))
print(X_train.shape[0])
print(np.shape(X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)))
print('标签：', y_train[0:20])

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)  # 把第三个通道1补上
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)  # 元组类型，元素不能更改

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# 构建模型
model = Sequential()


def add_conv(c):
    model.add(Convolution2D(c, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(c, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 池化层


def add_conv1(c):
    model.add(Convolution2D(c, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(c, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层
    model.add(Convolution2D(c, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 池化层


#conv1
model.add(Convolution2D(64, (kernel_size[0], kernel_size[1]), padding='same',
                        input_shape=input_shape))  # 卷积层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层
model.add(Convolution2D(64, (kernel_size[0], kernel_size[1]), padding='same'))  # 卷积层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))  # 池化层

#conv2
add_conv(128)
add_conv1(256)
add_conv1(512)
add_conv1(512)

#dense1
model.add(Flatten())  # 拉成一维数据
model.add(Dense(256))  # 全连接层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层

#dense1
model.add(Dense(256))  # 全连接层1
model.add(BatchNormalization())
model.add(Activation('relu'))  # 激活层

#dense1
model.add(Dense(10))  # 全连接层2
model.add(Activation('softmax'))  # Softmax评分
# sgdop = optimizers.gradient_descent_v2.SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])



import datetime
starttime = datetime.datetime.now()


# 训练模型
# model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=1, validation_data=(X_test, Y_test))

#long running
endtime = datetime.datetime.now()
print((endtime - starttime).seconds)
