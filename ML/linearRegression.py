import os, sys

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import SCORERS

data = np.array([[6], [8], [10], [14], [18]]).reshape(-1, 1)
label = [7, 9, 13, 17.5, 18]
class LR:
    def __init__(self, epoch=10) -> None:
        self.model = None
        self.epoch = epoch

    def fit(self, x, y):
        """
        x is colume vectors
        y is a list
        """
        self.model = LinearRegression()
        self.model.fit(x, y)
    
    def train(self, x, y):
        """
        x is colume vectors
        y is a list
        """
        n, m = x.shape
        xT = x.T
        xTx = xT.dot(x)
        if np.linalg.det(xTx) != 0.0:
            self.fit(x, y)
        else:
            omege = np.random.randn(m, 1)
            lr = 1e-3
            for i in range(self.epoch):
                err = x.dot(omege) - np.array(y).reshape(-1, 1)
                loss = np.mean(err**2)
                # print('loss is {}'.format(loss))
                # print(i)
                delta = 2 * xT.dot(err) / n
                omege -= lr * delta

            self.model = omege
    
    def predict(self, x):
        """
        x should be colume vector
        """
        if isinstance(self.model, LinearRegression):
            return self.model.predict(x)
        else:
            if isinstance(x, list):
                x = np.array(x).reshape(-1, self.model.shape[0])
            res = x.dot(self.model)
            res = np.squeeze(res)
            return res.tolist()
    
    def score(self, x, y):
        score = None
        if isinstance(self.model, LinearRegression):
            score = self.model.score(x, y)
        else:
            res = np.array(self.predict(x))
            if isinstance(y, list):
                y = np.array(y)
                mean = np.mean(y)
                ss_tot = np.mean((y - mean)**2)
                ss_res = np.mean((y - res)**2)
                score = float(1 - ss_res / ss_tot)
        return score

model = LR(epoch=100)
model.train(data, label)
res = model.predict([12])
x_test = [8, 9, 11, 16, 12]
y_test = [11, 8.5, 15, 18, 11]
res = model.predict(x_test)
print(res)
score = model.score(x_test, y_test)
print(score)