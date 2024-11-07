import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

class CustomClassifier:
    def __init__(self, lr, n_epoch, n_classes):
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_classes = n_classes
        self.w = None
        self.b = None

    def train(self, X, y):
        self.w = np.random.randn(X.shape[1], self.n_classes) * 0.01
        self.b = np.zeros((1, self.n_classes))
        for _ in range(self.n_epoch):
            grad_w, grad_b = self._grad(X, y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X):
        h = np.dot(X, self.w) + self.b
        h -= np.max(h, axis=1, keepdims=True)
        softmax = np.exp(h) / (np.sum(np.exp(h), axis=1, keepdims=True) + 1e-10)
        return softmax

    def _grad(self, X, y):
        y_pred = self.predict(X)
        delta = y_pred - y
        grad_w = np.dot(X.T, delta) / X.shape[0]
        grad_b = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
        return grad_w, grad_b
