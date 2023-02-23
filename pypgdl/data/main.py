import csv

import numpy as np
import pandas
import pandas as pd
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


n = 10000
size = 28*28*1
name = 'mnist'

print(np.shape(X_train), np.shape(y_train))
print(np.shape(X_test), np.shape(y_test))



x = np.reshape(X_test, (n, size))
y = y_test

empty = pd.DataFrame(columns=['a'])
for i in range(n):
    str1 = "{"
    for j in range(size):
        str1 = str1 + str(x[i][j]) + ","
    str1 = str1 + str(x[i][size-1]) + "}"
    new = pd.DataFrame({"a": str1}, index=["0"])
    empty = empty.append(new, ignore_index=True)
print(empty)
print(empty.shape)
empty.to_csv(name+'_x_test.csv',header=0,index=None)
y = pd.DataFrame(y)
y.to_csv(name+'_y_test.csv',header=0,index=None)
