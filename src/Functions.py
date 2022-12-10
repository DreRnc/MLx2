import numpy as np

def IdentityF(x):
    return x
class Identity():
    def __call__(self, x):
        return IdentityF(x)
    def derivative(self, x):
        return np.ones(x.shape)

def SigmoidF(x):
    return 1 / (1 + np.exp(-x))
class Sigmoid():
    def __call__(self, x):
        return SigmoidF(x)
    def derivative(self, x):
        return self(x) * (1 - self(x))

def ReLuF(x):
    return np.maximum(0, x)
class ReLU():
    def __call__(self, x):
        return ReLuF(x)
    def derivative(self, x):
        return (x > 0).astype(int)

class NoReg():
    def __call__(self, w):
        return 0
    def derivative(self, w):
        return np.zeros(w.shape)

class L1Reg():
    #L1 è la somma degli assoluti dei pesi (Lasso)
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))

class L2Reg():
    #L2 è la somma dei quadrati dei pesi (Ridge)
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    def __call__(self, w):
        return self.alpha * np.sum(np.square(w))

class ElasticReg():
    #E' la somma di L1 e L2
    def __init__(self, alpha_l1=0.1, alpha_l2=0.1):
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2
    def __call__(self, w):
        return self.alpha_l1 * np.sum(np.abs(w)) + self.alpha_l2 * np.sum(np.square(w))

def MSE(y, y_pred):
    if y.shape != y_pred.shape:
        raise ValueError("inputs must have the same shape")
    return np.mean(np.square(y - y_pred))

def MAE(y, y_pred):
    if y.shape != y_pred.shape:
        raise ValueError("inputs must have the same shape")
    return np.mean(np.abs(y - y_pred))

def Accuracy(y, y_pred):
    if y.shape != y_pred.shape:
        raise ValueError("inputs must have the same shape")
    if y.shape[1]>1:
        y = np.argmax(y, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y == y_pred)
















