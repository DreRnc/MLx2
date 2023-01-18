import numpy as np
from sklearn.utils import shuffle
import random
import math
import matplotlib.pyplot as plt


class SVM:
    def __init__(self, X : np.array, y: np.array,
            C=10000, l=0.000001, max_ep=5000,
            c_th=0.01, visualization=True,
            kernel="rbf", gammaMode = "scale", degree = 3, coef0 = 0) -> None:
        # features, output and weights
        self.n = X.shape[1]
        self.X = X
        self.y = y
        self.W = np.zeros(self.n)

        # hyperparameters
        self.C = C
        self.l = l

        # kernel
        self.kernel = kernel # linear, rbf, poly, sigmoid
        self.degree = degree # poly
        self.gamma = 1 / self.n if gammaMode == "auto" else 1 / (self.n * X.var())  # auto, scale
        self.coef0 = coef0

        self.max_ep = max_ep
        self.c_th = c_th

        self.last_cost = float("inf")

        if (visualization):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def train(self) -> np.array:
        
        w = self.W
        for epoch in range(1, int(self.max_ep)):
            X, y = shuffle(X, y)
            for ind, x in enumerate(X):
                w += - self.l * self.get_dw(w, x, y[ind])
            if (epoch & (epoch - 1) == 0) or (epoch - 1 == self.max_ep):
                cost = self.cost(w, X, y)
                print(f"Epoch: {epoch}, Cost: {cost}")
                self.W = w

                if abs(self.last_cost - cost) < self.c_th * self.last_cost:
                    return w
                # self.visualize()
                self.last_cost = cost if cost < self.last_cost else self.last_cost
        self.W = w
        return w

    def cost(self, w, x, y):
        dist = self.distance(w, x, y)
        dist[dist < 0] = 0
        return 1/2 * self.dot(w, w) + self.C * np.average(dist)

    def get_dw(self, w: np.array, x, y) -> float:
        return - self.C * y * x if self.distance(w, x, y) > 0 else 0

    def distance(self, w, x, y):
        return 1 - y * (self.dot(x, w))

    def visualize(self):
        def hp(x, w, b, v):
            return (- w[0] * x - b + v) / w[1]

        for ind, x in enumerate(self.X):
            if self.y[ind] == 1:
                self.ax.scatter(x[0], x[1], s=100, marker='+', color='red')
            else:
                self.ax.scatter(x[0], x[1], s=100, marker='_', color='blue')
        range = [-1, 10]
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], 1)
                     for x in range], color='black')
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], -1)
                     for x in range], color='black')
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], 0)
                     for x in range], color='green')
        plt.show()

    def dot(self, x, y):
        kernel = self.kernel
        if kernel == "rbf":
            return math.exp(-self.gamma * (np.dot(x, y) ** 2))
        elif kernel == "poly":
            return (self.gamma * np.dot(x, y) + self.coef0) ** self.degree
        elif kernel == "sigmoid":
            return math.tanh(self.gamma * np.dot(x, y) + self.coef0)
        else:  # kernel == "linear"
            return np.dot(x, y)
        
def main():
    w = np.array([1, 1])
    X = []
    y = []
    nentries = 100
    for i in range(nentries):
        entry = []
        dot = 0
        for coord in [0, 1]:
            x = random.randint(0, 10)
            entry.append(x)
            dot += x * w[coord]
        if (random.randint(0, 100) > 95):
            dot = -dot
        entry.append(1)
        X.append(np.array(entry))
        y.append(1 if dot > 10 else -1)
    print(X)
    print(y)
    X = np.array(X)
    y = np.array(y)

    svm = SVM(max_ep=1e9, c_th=1e-2)
    w_trained = svm.train(X, y)

    # print(w_trained)
    # svm.visualize()


if __name__ == "__main__":
    main()
