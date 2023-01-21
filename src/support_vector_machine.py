import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle


# >> FEATURE SELECTION << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    # for itr in range(0, len(X.columns)):
    #     regression_ols = sm.OLS(Y, X).fit()
    #     max_col = regression_ols.pvalues.idxmax()
    #     max_val = regression_ols.pvalues.max()
    #     if max_val > sl:
    #         X.drop(max_col, axis='columns', inplace=True)
    #         columns_dropped = np.append(columns_dropped, [max_col])
    #     else:
    #         break
    regression_ols.summary()
    return columns_dropped


##############################


# >> MODEL TRAINING << #
def compute_cost(w, x, y):
    # calculate hinge loss
    N = x.shape[0]
    distances = 1 - y * (np.dot(x, w))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = regularization_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(w, w) + hinge_loss
    return cost


# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])  # gives multidimensional array

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        dw += di

    dw = dw/len(Y_batch)  # average
    return dw


def sgd(features, outputs):
    max_epochs = 5000
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.01  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = shuffle(features, outputs)
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)

        # convergence check on 2^nth epoch
        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1
    return weights


########################


def init(data_path):
    print("reading dataset...")
    # read data in pandas (pd) data frame
    data = pd.read_csv(data_path)

    # drop last column (extra column added by pd)
    # and unnecessary first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)

    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]

    # filter features
    remove_correlated_features(X)
    remove_less_significant_features(X, Y)

    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(
        X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # testing the model
    print("testing the model...")
    y_train_predicted = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(X_train.to_numpy()[i], W))
        y_train_predicted = np.append(y_train_predicted, yp)

    y_test_predicted = np.array([])
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(X_test.to_numpy()[i], W))
        y_test_predicted = np.append(y_test_predicted, yp)

    print("accuracy on test dataset: {}".format(
        accuracy_score(y_test, y_test_predicted)))
    print("recall on test dataset: {}".format(
        recall_score(y_test, y_test_predicted)))
    print("precision on test dataset: {}".format(
        recall_score(y_test, y_test_predicted)))


# set hyper-parameters and call init
regularization_strength = 10000
learning_rate = 0.000001
init()


class SVM:
    def __init__(self, C=10000, l=0.000001, max_ep=5000, c_th=0.01) -> None:
        self.X = None
        self.y = None
        self.W = None
        self.n = None
        self.C = C
        self.l = l
        self.max_ep = max_ep
        self.c_th = c_th
        self.last_cost = float("inf")

    def train(self, X, y):
        w = np.zeros(X.shape[1])
        for epoch in range(1, self.max_ep):
            X, y = shuffle(X, y)
            for ind, x in enumerate(X):
                w += - self.l * self.get_dw(w, x, y[ind])
            if (epoch & (epoch - 1) == 0) or (epoch - 1 == self.max_ep):
                cost = self.cost(w, X, y)
                print(f"Epoch: {epoch}, Cost: {cost}")
                if abs(self.last_cost - cost) < self.c_th * self.last_cost:
                    return w
                self.last_cost = cost
        return w

    def cost(self, w, x, y):
        dist = 1 - y * (np.dot(x, w))
        dist[dist < 0] = 0
        return 1/2 * np.dot(w, w) + self.C/self.n * np.sum(dist)

    def get_dw(self, w: np.array, x, y) -> float:
        dist = 1 - y * (np.dot(x, w))
        return - self.C * y * x if np.dot > 0 else 0