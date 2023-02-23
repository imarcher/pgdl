
def run(batch, epochs):
    import numpy as np
    np.random.seed(1337)  # for reproducibility
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
    from keras.layers import Convolution2D, MaxPooling2D
    from keras.utils import np_utils
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 全局变量
    batch_size = batch
    nb_classes = 2
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

    for j in range(epochs):
        for num in range(40):
            X_train = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__x_train.npy')
            y_train = np.load('/home/hh/data/keras_epsilon/' + str(num) + name + '__y_train.npy')
            # 转换为one_hot类型
            Y_train = np_utils.to_categorical(y_train, nb_classes)
            # 训练模型
            model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)






