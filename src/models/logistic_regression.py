import numpy as np

class LogisticRegression:

    def __init__(self, lr: float = 0.01, epoch: int = 100, fit_intercept: bool = True, verbose: bool = False):
        self.lr = lr
        self.epoch = epoch
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X: np.ndarray) -> np.ndarray:
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def __loss(self, h: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -y * np.log(h) - (1 - y) * np.log(1 - h)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.epoch):
            logit = np.dot(X, self.theta)
            h = self.__sigmoid(logit)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        if self.fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return self.predict_prob(X) > threshold