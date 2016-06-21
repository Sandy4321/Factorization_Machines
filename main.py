__author__ = 'zhuchen02'

import numpy as np

import fm

if __name__ == '__main__':
    fm_ = fm.Fm(5, 'c')
    train_x = []
    train_y = []
    test = []
    with open('data/iris/iris.data', 'rb') as f:
        for line in f:
            s = line.strip().split(',')
            if s[-1] == 'Iris-setosa':
                train_x.append([float(i) for i in s[:-1]])
                train_y.append(1)
            if s[-1] == 'Iris-versicolor':
                train_x.append([float(i) for i in s[:-1]])
                train_y.append(-1)
            if s[-1] == 'Iris-virginica':
                test.append([float(i) for i in s[:-1]])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test = np.array(test)

    fm_.learn(train_x, train_y)
    print fm_.w_0
    print fm_.w
    print fm_.v
    print fm_.predict(train_x)
    fm_.save('tmp')
    fm_1 = fm.Fm(5, 'c')
    fm_1.load('tmp')
    print fm_1.predict(train_x)