__author__ = 'zhuchen02'

import numpy as np
import codecs

class Fm:
    k = -1
    task = -1
    iter = 500
    threshold = 0.0001
    n = -1
    learning_rate = 0.0001

    w_0 = None
    w = None
    v = None

    def __init__(self, k, task):
        if not (type(k) == int and k > 0):
            raise
        self.k = k
        if not task not in [0, 1]:
            raise
        self.task = task

    def load(self, addr):
        '''
        input model from mode file
        :param addr: the addr of model file
        :return: none
        '''
        with codecs.open(addr, 'rb', 'utf-8') as f:
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'k':
                raise Exception('formation of input file is wrong. k')
            else:
                self.k = int(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'task' or s[1] not in ['0', '1']:
                raise Exception('formation of input file is wrong. task')
            else:
                self.task = int(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'iter':
                raise Exception('formation of input file is wrong. iter')
            else:
                self.iter = int(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'threshold':
                raise Exception('formation of input file is wrong. threshold')
            else:
                self.threshold = float(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'n':
                raise Exception('formation of input file is wrong. n')
            else:
                self.k = int(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'learning_rate':
                raise Exception('formation of input file is wrong. learning_rate')
            else:
                self.learning_rate = float(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'w_0':
                raise Exception('formation of input file is wrong. w_0')
            else:
                self.w_0 = float(s[1])
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'w':
                raise Exception('formation of input file is wrong. w')
            else:
                self.w = np.array([float(w) for w in s[1].split(' ')])
                if len(self.w) != self.n:
                    raise Exception('formation of input file is wrong. w')
            s = f.readline().encode('utf-8').strip().split(':')
            if len(s) != 2 or s[0] != 'v':
                raise Exception('formation of input file is wrong. v')
            else:
                tmp = np.array([float(w) for w in s[1].split(' ')])
                if len(tmp) != self.n*self.k:
                    raise Exception('formation of input file is wrong. v')
                else:
                    self.v = tmp.resize(self.n, self.k)
            if self.task == 1:
                s = f.readline().encode('utf-8').strip().split(':')
                if len(s) != 2 or s[0] != 'max':
                    raise Exception('formation of input file is wrong. max')
                else:
                    self.max = float(s[1])
                s = f.readline().encode('utf-8').strip().split(':')
                if len(s) != 2 or s[0] != 'min':
                    raise Exception('formation of input file is wrong. min')
                else:
                    self.min = float(s[1])

        pass

    def save(self, addr):
        '''
        output model to model file
        :param addr: the addr of model file
        :return: none
        '''
        with codecs.open(addr, 'wb', 'utf-8') as f:
            f.write('k:'+str(self.k))
            f.write('task:'+str(self.task))
            f.write('iter:'+str(self.iter))
            f.write('threshold:'+str(self.threshold))
            f.write('n:'+str(self.n))
            f.write('learning_rate:'+str(self.learning_rate))
            f.write('w_0:'+str(self.w_0))
            f.write('w:'+' '.join([str(w) for w in self.w]))
            f.write('v:'+' '.join([str(v) for v in self.np.reshape(self.v, (1, self.n*self.k))[0]]))
            if self.task == 1:
                f.write('max:'+str(self.max))
                f.write('min:'+str(self.min))

    def learn(self, x_s, y_s):
        '''

        :return:
        '''
        if len(x_s) != len(y_s):
            raise Exception('')
        if type(x_s) == list:
            x_len = -1
            for i in x_s:
                if x_len != -1 and x_len != len(i):
                    raise Exception('')
                x_len = len(i)
                for j in i:
                    if type(j) not in [int, float]:
                        raise Exception('')
            x_s = np.array(x_s)
        if type(y_s) == list:
            for i in y_s:
                if type(i) not in [int, float]:
                    raise Exception('')
            y_s = np.array(y_s)
        if type(x_s) != np.ndarray or type(y_s) != np.ndarray:
            raise Exception('')

        self.n = x_s.shape[1]
        self.w_0 = 0.0
        self.w = np.zeros(self.n)
        self.v = np.random.normal(0, 0.01, (self.n, self.k))
        if self.task == 0:
            self._learn_classify(x_s, y_s)
        elif self.task == 1:
            self._learn_regress(x_s, y_s)

    def _learn_classify(self, x_s, y_s):
        for i in range(self.iter):
            for x, y in zip(x_s, y_s):
                y_p = self._predict(x)
                mult = -y*(1.0-1.0/(1+np.exp(y*y_p)))
                self.w_0 -= self.learning_rate*mult
                for x_iter in range(self.n):
                    self.w[x_iter] -= self.learning_rate*mult*x[x_iter]
                for x_iter in range(self.n):
                    for k_iter in range(self.k):
                        self.v[x_iter, k_iter] -= self.learning_rate*mult*\
                                                  (x[x_iter]*np.dot(self.v[:, k_iter], x)-self.v[x_iter, k_iter]*x[x_iter]**2)

    def _learn_regress(self, x_s, y_s):
        self.max = np.max(y_s)
        self.min = np.min(y_s)
        for i in range(self.iter):
            for x, y in zip(x_s, y_s):
                y_p = self._predict(x)
                mult = -(y-y_p)
                self.w_0 -= self.learning_rate*mult
                for x_iter in range(self.n):
                    self.w[x_iter] -= self.learning_rate*mult*x[x_iter]
                for x_iter in range(self.n):
                    for k_iter in range(self.k):
                        self.v[x_iter, k_iter] -= self.learning_rate*mult*\
                                                  (x[x_iter]*np.dot(self.v[:, k_iter], x)-self.v[x_iter, k_iter]*x[x_iter]**2)

    def predict(self, x_s):
        '''

        :param x_s:
        :return:
        '''
        return [self._predict(x) for x in x_s]

    def _predict(self, x):
        '''

        :param x:
        :return:
        '''
        self._data_validate(x)
        if self.task == 0:
            return self._predict_classify(x)
        elif self.task == 1:
            return self._predict_regress(x)

    def _predict_classify(self, x):
        y_p = self.w_0 + np.dot(self.w, x)
        for k_iter in range(self.k):
            y_p += 0.5*np.dot(self.v[:, k_iter], x)**2-\
                   0.5*np.dot(self.v[:, k_iter]**2, x**2)
        y_p = 1.0/(1.0+np.exp(-y_p))
        return y_p

    def _predict_regress(self, x):
        y_p = self.w_0 + np.dot(self.w, x)
        for k_iter in range(self.k):
            y_p += 0.5*np.dot(self.v[:, k_iter], x)**2-\
                   0.5*np.dot(self.v[:, k_iter]**2, x**2)
        y_p = np.min(self.max_y, y_p)
        y_p = np.max(self.min_y, y_p)
        return y_p

    def _data_validate(self, x):
        if len(x) != self.n:
            raise Exception('formation of x is wrong')