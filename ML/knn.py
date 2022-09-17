import os, sys
from typing import List

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer

data = np.array([
    [158, 64],
    [170, 86],
    [183, 84],
    [191, 80],
    [155, 49],
    [163, 59],
    [180, 67],
    [158, 54],
    [170, 67]
])
label = ['male', 'male', 'male', 'male','female', 'female', 'female', 'female', 'female']

class KNN:
    def __init__(self, k, flavor='sklearn') -> None:
        if flavor == 'sklearn':
            self.model = KNeighborsClassifier(n_neighbors=k)
            self.transformer = LabelBinarizer()
        else:
            self.model = k
            self.data = None
            self.label = None
        
    def fit(self, x:np.ndarray, y:List):
        self.label = y
        if isinstance(self.model, KNeighborsClassifier):
            y = self.transformer.fit_transform(y)
            self.model.fit(x, y.reshape(-1))
        else:
            self.data = x
    
    def native(self, x):
        dis = x[:, :, np.newaxis] - self.data.transpose(1, 0)[np.newaxis, :, :]
        dis = np.sum(dis**2, axis=1)
        index = dis.argsort(axis=1)[:, :self.model]
        res = []
        for ind in index:
            labels = np.take(self.label, ind)
            cnt = Counter(labels)
            res.append(cnt.most_common(1)[0][0])
        return res

    def predict(self, x):
        if isinstance(self.model, KNeighborsClassifier):
            res_binary = self.model.predict(x)
            res = self.transformer.inverse_transform(res_binary).tolist()
        else:
            res = self.native(x)
        return res

model = KNN(3, flavor='common')
model.fit(data, label)
sample = np.array([[155, 70]])
res = model.predict(sample)
x_test = np.array([
    [168, 65],
    [180, 96],
    [160, 52],
    [169, 67]
])
res = model.predict(x_test)
print(res)
