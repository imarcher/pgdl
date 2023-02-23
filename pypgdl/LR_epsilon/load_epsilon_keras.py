from pandas import Series,DataFrame
import pandas as pd
import re
import numpy as np

# result = pd.read_pickle('../epsilon_normalized.t/epsilon_normalized.t',)
# print(result.shape)
name = 'epsilon_'
n = 10000
size = 2000

for num in range(40):
    print(num)
    result = pd.read_csv('../epsilon_normalized/epsilon_normalized.csv',header=None, skiprows=n*num, nrows=n)
    print(result.shape)
    print(result[0][0])

    y = np.zeros((n,1))
    x = np.zeros((n,2000))
    yi = 0
    for i in range(n):
        linelist = str.split(result[0][i]," ")
        # print(len(linelist))
        # #y
        if linelist[0] == '-1':
            y[yi][0] = 0
        else:
            y[yi][0] = linelist[0]
        for j in range(1,size):
            matchObj = re.match( r'(.*):(.*)', linelist[j], re.M|re.I)
            x[yi][j-1] = matchObj.group(2)
        matchObj = re.match(r'(.*):(.*)', linelist[size], re.M | re.I)
        x[yi][size-1] = matchObj.group(2)
        yi = yi+1


    np.save('keras_epsilon/'+str(num)+name+'_x_train',x)
    np.save('keras_epsilon/'+str(num)+name+'_y_train',y)