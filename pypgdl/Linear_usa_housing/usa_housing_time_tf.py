def run(batch, epochs):
    import numpy as np
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # 全局变量
    batch_size = batch

    model = Sequential()
    model.add(Dense(32))  # 全连接层1
    model.add(BatchNormalization())
    model.add(Activation('relu'))  # 激活层

    model.add(Dense(1))  # 全连接层3

    model.compile(loss='mse',
                  optimizer="sgd",
                  metrics=['mse'])

    X_train = np.load('/home/hh/pypgdl/data/USA_Housing/USA_Housing_x_train_tf.npy')
    Y_train = np.load('/home/hh/pypgdl/data/USA_Housing/USA_Housing_y_train_tf.npy')

    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
