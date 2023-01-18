import numpy as np
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
class SVM:
    def __init__(self, C=10000, l=0.000001, max_ep=5000, c_th=0.01, visualization = True) -> None:
        self.X = None
        self.y = None
        self.W = None
        self.n = None
        self.C = C
        self.l = l
        self.max_ep = max_ep
        self.c_th = c_th
        self.last_cost = float("inf")
        if(visualization):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def train(self, X, y) -> np.array:
        self.X = X
        self.y = y
        w = np.zeros(X.shape[1])
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
        return w

    def cost(self, w, x, y):
        dist = 1 - y * (np.dot(x, w))
        dist[dist < 0] = 0
        return 1/2 * np.dot(w, w) + self.C * np.average(dist)

    def get_dw(self, w: np.array, x, y) -> float:
        dist = 1 - y * (np.dot(x, w))
        return - self.C * y * x if dist > 0 else 0

    def visualize(self):
        def hp(x, w, b, v):
            return (- w[0] * x - b + v) / w[1]
        
        for ind, x in enumerate(self.X):
            if self.y[ind] == 1:
                self.ax.scatter(x[0], x[1], s=100, marker='+', color='red')
            else:
                self.ax.scatter(x[0], x[1], s=100, marker='_', color='blue')
        range = [-1, 10]
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], 1) for x in range], color='black')
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], -1) for x in range], color='black')
        self.ax.plot(range, [hp(x, self.W[:-1], self.W[-1], 0) for x in range], color='green')
        plt.show()


def main():
    w = np.array([1, 1])
    X = []
    y = []
    nentries = 100
    for i in range(nentries):
        entry = []
        dot = 0
        for coord in [0, 1]:
            x =  random.randint(0, 10)
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

    print(w_trained)
    svm.visualize()


if __name__ == "__main__":
    main()
    