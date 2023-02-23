import numpy as np
import time
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(np.shape(X_train), np.shape(y_train))
print(np.shape(X_test), np.shape(y_test))

