import numpy as np
from src.MetricFunctions import get_metric_instance
import numpy as np

def step(x):
    """
    Returns 1 if x > 0, 0 otherwise
    """
    return 1 if x >= 0 else -1

class SVR:

    def __init__ (self, verbose = False):
        """
        Attributes
        ----------
        X: training set
        y: labels of training set
        K: kernel matrix
        lambdas: vector of lambdas
        b: bias
        errors: vector of errors
        gamma: parameter of rbf kernel
        C: regularization parameter
        offset: parameter of polynomial kernel
        degree: parameter of polynomial kernel
        epsilon: parameter of epsilon-insensitive loss
        tol: tolerance for stopping criterion
        iter: number of iterations
        kernel: kernel function
        verbose: if True, prints information about the training process

        """
        self.verbose = verbose
        # print("X: ", X.shape, "y: ", y.shape, "K: ", self.K.shape, "lambdas: ", self.lambdas.shape, "errors: ", self.errors.shape)

    def calculate_kernel (self, x1, x2):
        """
        x1, x2 points of training or test set
        Shape = [n_features x 1]
        """

        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (np.dot(x1, x2) + self.offset)**self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)
        else:
            raise ValueError('Invalid kernel function')


    def generate_kernel_matrix (self, X): 
        """
        Returns the kernel matrix for the given training data.

        X training set
        Shape = [n__training_samples x n_features]

        K[i, j] = K(x_i, x_j) for the X points in training set.
        Shape = [n_training_samples x n_training_samples]
        """
        # print("Training set: ", X.shape, "should be (n_samples, n_features)")
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = self.calculate_kernel(X[i], X[j])
        # print("Kernel matrix: ", K.shape, "should be (n_samples, n_samples)")
        return K
        
    
    def compute_kernel_vector (self, test_sample):
        """
        Returns the kernel vector for the given test sample

        x test sample
        Shape = [1 x n_features]

        k kernel vector
        Shape = [n_training_samples x 1]

        """
        
        k = np.zeros((self.X.shape[0], 1))
        for i in range(self.X.shape[0]):
            k[i] = self.calculate_kernel(self.X[i], test_sample)
        # print("Kernel vector: ", k.shape, "should be (n_samples, 1)")

        return k
    
    def predict (self, X_p):
        """
        Returns a vector of predictions for set X_p
        
        X_p is a set of samples
        Shape = [n_samples x n_features]
        
        pred is the vector of predictions
        Shape = [n_samples x 1]
        
        """

        pred = np.zeros(X_p.shape[0])
        for i in range(X_p.shape[0]):
            k = self.compute_kernel_vector(X_p[i])
            pred[i] = np.dot(self.lambdas, k) + self.b
        # print('Predictions:', pred.shape , X_p.shape[0])
        return pred
    
    def fit (self, X, y, kernel, C, gamma, offset, degree, epsilon, tolerance, max_iter):
        """
        Creates SVR model resolving dual problem with SMO solver.
        """
        self.X = X
        self.y = y

        self.gamma = gamma
        self.C = C
        self.offset = offset 
        self.degree = degree
        self.epsilon = epsilon
        self.tol = tolerance

        self.kernel = kernel
        self.K = self.generate_kernel_matrix(X)
        self.lambdas = np.zeros(X.shape[0])
        self.b = 0
        self.errors = np.zeros(X.shape[0])


        num_changed = 0
        examine_all = True
        self.errors = np.copy(-self.y)
        self.iter = 0

        while num_changed > 0 or examine_all:
            num_changed = 0
            lambdas_old = np.copy(self.lambdas)

            if examine_all:
                for i in range(self.X.shape[0]):
                    num_changed_f = self.examine_example(i)
                    num_changed += num_changed_f
            else:
                for i in range(self.X.shape[0]):
                    if self.lambdas[i] != 0 and self.lambdas[i] != self.C:
                        num_changed_f = self.examine_example(i)
                        num_changed += num_changed_f
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            self.iter += 1

            if self.verbose:
                print(f'Iteration: {self.iter}, num_changed: {num_changed}, norm: {np.linalg.norm(self.lambdas - lambdas_old)}')
            # print("Old lambdas: ", lambdas_old)
            # print("New lambdas: ", self.lambdas)


            if np.linalg.norm(self.lambdas - lambdas_old) < self.tol or self.iter >= max_iter:
                if self.verbose:
                    print(f'Max iterationn reached: {self.iter}')
                break

    
    def examine_example (self, i2):
        """
        """
        num_changed_f = 0

        if ((self.errors[i2] < -self.tol) and (self.lambdas[i2] < self.C)) or ((self.errors[i2] > self.tol) and (self.lambdas[i2] > 0)) or (self.iter == 0):
            idx = np.where((self.lambdas != 0) & (self.lambdas != self.C))[0]
            if len(idx) > 1:
                if self.errors[i2] > 0:
                    i1 = np.argmin(self.errors)
                else:
                    i1 = np.argmax(self.errors)
                if i1 != i2:
                    self.take_step(i1, i2)
                    num_changed_f += 1
            if num_changed_f == 0:
                idx = np.random.permutation(idx)
                for i1 in idx:
                    if i1 != i2:
                        self.take_step(i1, i2)
                        num_changed_f += 1
                        break
            if num_changed_f == 0:
                idx = np.random.permutation(self.lambdas.shape[0])
                for i1 in idx:
                    if i1 != i2:
                        self.take_step(i1, i2)
                        num_changed_f += 1
                        break

        return num_changed_f

    def take_step (self, i1, i2):
        """
        Following algorithm in table 1 from paper by G.W. Flake.
        """
        y1 = self.y[i1]
        y2 = self.y[i2]
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]

        #Save old values (lambdas, y and function evaluations)
        b_old =  np.copy(self.b)
        l1_old = np.copy(self.lambdas[i1])
        l2_old = np.copy(self.lambdas[i2])
        f1_old = self.K[i1, :] @ self.lambdas + self.b
        f2_old = self.K[i2, :] @ self.lambdas + self.b

        # Steps 1 to 15 of algorithm in table 1 from paper by G.W. Flake.
        s = l1_old + l2_old

        eta = k11 + k22 - 2 * k12

        delta = 2 * self.epsilon / eta

        self.lambdas[i1] = l1_old + (1 / eta) * (y1 - y2 - f1_old + f2_old)
        self.lambdas[i2] = s - self.lambdas[i1]

        if self.lambdas[i1] * self.lambdas[i2] < 0:
            if np.abs(self.lambdas[i1]) >= 0 and np.abs(self.lambdas[i2]) >= 0:
                self.lambdas[i1] = self.lambdas[i1] - np.sign(self.lambdas[i1]) * delta
            else:
                self.lambdas[i1] = step(np.abs(self.lambdas[i1]) - np.abs(self.lambdas[i2])) * s
        
        L = max(s - self.C, - self.C)
        H = min(self.C, s + self.C)

        self.lambdas[i1] = min(H, max(L, self.lambdas[i1]))
        self.lambdas[i2] = s - self.lambdas[i1]

        # Update the threshold b to reflect change in Lagrange multipliers
        # Forces f1 = y1 or f2 = y2
        # If both are not satisfied, then the threshold is updated to the average of the two

        b1 = y1 - f1_old + (l1_old - self.lambdas[i1]) * k11 + (l2_old - self.lambdas[i2]) * k12 + b_old
        b2 = y2 - f2_old + (l1_old - self.lambdas[i1]) * k12 + (l2_old - self.lambdas[i2]) * k22 + b_old
        if 0 < self.lambdas[i1] < self.C:
            b = b1
        elif 0 < self.lambdas[i2] < self.C:
            b = b2
        else:
            b = (b1 + b2) / 2
        
        self.b = b

        # Update error cache using new Lagrange multipliers
        for i in range(self.X.shape[0]):
            self.errors[i] = self.errors[i] + (self.lambdas[i1] - l1_old) * self.K[i1, i] + (self.lambdas[i2] - l2_old) * self.K[i2, i] + b - b_old

        return

    
class MultiOutputSVR:
    """
    Multi-output SVM regression.
    """

    def __init__(self, n_outputs = 2, kernel = 'rbf', verbose = False):
        """
        Initialize the SVR.
        """
        self.task = 'regression'
        self.kernel = kernel
        self.verbose = verbose

        self.models = [SVR(verbose = verbose) for _ in range(n_outputs)]
        
    def fit(self, X, y, C = 0.1, gamma = 'auto', offset = 1, degree = 3, epsilon = 0.1, tolerance = 0.001, max_iter = 1000):
        """
        Fit the SVM to the provided data with SMO algorithm.
        """
        for i, model in enumerate(self.models):
            if self.verbose:
                print(f'Target {i + 1} training')

            model.fit(X, y[:, i], kernel = self.kernel, C = C, gamma = gamma, offset = offset, \
                degree = degree, epsilon = epsilon, tolerance = tolerance, max_iter = max_iter)

            if self.verbose:
                print(f'Target {i + 1} trained')

    def evaluate_model(self, X, y, metric = 'mee'):
        """
        Test the SVM on the provided data.
        """
        metric = get_metric_instance(metric)
        y_pred = self.predict(X)
        return metric(y, y_pred)
    
    def predict(self, X):
        """
        Predict the target values for the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data to predict.

        Returns
        -------
        array-like, shape = [n_samples, n_outputs]
            The predicted target values.
        
        """

        y_pred = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            y_pred[:, i] = model.predict(X)
        return y_pred   
