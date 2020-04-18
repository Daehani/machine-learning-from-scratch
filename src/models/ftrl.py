# https://www.kaggle.com/jiweiliu/ftrl-starter-code

import numpy as np
from math import exp, log, sqrt
from random import random
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union

class FtrlProximal():

    def __init__(self, feature_size: int, bias: bool, alpha: float, beta: float, l1: float, l2: float, seed: int = 1234) -> None:
        # parameters
        self.bias = bias
        self.alpha = alpha
        self.beta = beta
        self.L1 = l1
        self.L2 = l2

        # # feature related parameters
        # self.D = D

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        if self.bias is True:
            feature_size += 1
        self.n = [0.] * feature_size
        np.random.seed(seed)
        self.z = np.random.rand(feature_size)
        self.w = {}

    def _indices(self, x: Dict[int, float]) -> int:
        # normal indices
        for index in x.keys():
            yield index

    def _data_generator(self, X_path: str, y_path: Optional[str] = None) -> Union[Tuple[int, Dict[int, float]], Tuple[int, Dict[int, float], int]]:
        def x_to_sparse(x: List[float], bias: bool = True) -> Dict[int, float]:
            if bias is True:
                x = [1] + x
            x_sparse = {}
            for i in range(len(x)):
                if x[i] != 0:
                    x_sparse[i] = x[i]
            return x_sparse

        if y_path is None:
            with open(X_path) as X_file:
                for t, row in enumerate(X_file):
                    x = [float(value) for value in row.strip().split(",")]
                    x = x_to_sparse(x, self.bias)
                    yield t, x
        else:
            with open(X_path) as X_file, open(y_path) as y_file:
                for t, (row, y) in enumerate(zip(X_file, y_file)):
                    x = [float(value) for value in row.strip().split(",")]
                    x = x_to_sparse(x, self.bias)
                    y = int(y)
                    yield t, x, y

    def _predict(self, x: Dict[int, float], is_train: bool) -> float:
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2
        n = self.n
        z = self.z
        if is_train is True:
            w = {}
        else:
            w = self.w.copy()

        wTx = 0.    # inner product of w and x
        for i in self._indices(x):
            if z[i] < 0:
                sign = -1.
            else:
                sign = 1.
            if sign * z[i] <= L1:
                w[i] = 0
            else:
                w[i] = -(1 / ((beta + sqrt(n[i])) / alpha + L2)) * (z[i] - sign * L1)

            wTx += w[i] * x[i]
        # cache the current w for update stage
        if is_train is True:
            self.w = w
        
        # bounded sigmoid function, this is the probability estimation
        score = 1. / (1. + exp(wTx))
        return score

    def _update(self, x: Dict[int, float], p: float, y: int) -> None:
        alpha = self.alpha
        n = self.n
        z = self.z
        w = self.w
        for i in self._indices(x):
            g = (p - y) * x[i]
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g

    def fit_generator(self, X_path: str, y_path: str, epoch_size: int) -> None:
        for e in range(epoch_size):
            loss = 0.
            count = 0
            for t, x, y in self._data_generator(X_path, y_path):
                p = self._predict(x, is_train=True)
                loss = logloss(p, y)
                # print process
                self._update(x, p, y)
                count += 1
                if count % 100 == 0:
                    print("{}\tepoch: {}\tcount: {}\tlogloss: {}".format(datetime.now(), e, count, loss/count))

    def predict_proba_generator(self, X_path: str, y_path: str) -> List[float]:
        w = self.w.copy()
        loss = 0.
        scores = []
        count = 0
        # for t, x in self._data_generator(X_path):
        for t, x, y in self._data_generator(X_path, y_path):
            p = self._predict(x, is_train=False)
            scores.append(p)
            loss += logloss(p, y)
            # print process
            count += 1
            if count % 100 == 0:
                print("{}\tepoch: {}\tcount: {}\tlogloss: {}".format(datetime.now(), e, count, loss/count))            
        return scores

def logloss(p: float, y: int) -> float:
    p = max(min(p, 1. - 10e-15), 10e-15)
    if y == 1:
        return -log(p)
    else:
        return -log(1. - p)