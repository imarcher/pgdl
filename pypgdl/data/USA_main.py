import csv

import numpy as np
import pandas
import pandas as pd

name = 'USA_Housing'

def nor(da):
    _range = np.max(da) - np.min(da)
    return (da - np.min(da)) / _range


data = pd.read_csv("USA_Housing/USA_Housing.csv", header=None)
print(data.shape)
print(data)

x = np.zeros((5000, 6))
for i in range(6):
    datanp = np.zeros((5000,))
    for j in range(5000):
        datanp[j] = data[i][j]
    datanp = nor(datanp)
    x[:, i] = datanp

print(x)

np.random.shuffle(x)
print(x)

xnp = x[:, 0:5]
ynp = x[:, 5]
ynp = np.reshape(ynp, (5000, 1))

xnp1 = xnp[0:4000, :]
ynp1 = ynp[0:4000, :]
xnp2 = xnp[4000:5000, :]
ynp2 = ynp[4000:5000, :]

np.save('USA_Housing_x_train_tf', xnp1)
np.save('USA_Housing_y_train_tf', ynp1)
np.save('USA_Housing_x_test_tf', xnp2)
np.save('USA_Housing_y_test_tf', ynp2)


empty = pd.DataFrame(columns=['a'])
for i in range(4000):
    str1 = "{"
    for j in range(4):
        str1 = str1 + str(xnp1[i][j]) + ","
    str1 = str1 + str(xnp1[i][4]) + "}"
    new = pd.DataFrame({"a": str1}, index=["0"])
    empty = empty.append(new, ignore_index=True)
empty.to_csv(name+'_x_train.csv',header=0,index=None)
y = pd.DataFrame(ynp1)
y.to_csv(name+'_y_train.csv',header=0,index=None)

empty = pd.DataFrame(columns=['a'])
for i in range(1000):
    str1 = "{"
    for j in range(4):
        str1 = str1 + str(xnp2[i][j]) + ","
    str1 = str1 + str(xnp2[i][4]) + "}"
    new = pd.DataFrame({"a": str1}, index=["0"])
    empty = empty.append(new, ignore_index=True)
empty.to_csv(name+'_x_test.csv',header=0,index=None)
y = pd.DataFrame(ynp2)
y.to_csv(name+'_y_test.csv',header=0,index=None)


# empty.to_csv(name+'x_test.csv',header=0,index=None)
# y = pd.DataFrame(y)
# y.to_csv(name+'y_test.csv',header=0,index=None)
