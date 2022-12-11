import numpy as np
import math

from Layers import Dense, FullyConnected
from Optimization_algorithms import gradient_descent
from Functions import MSE, NoReg,  L1Reg, L2Reg

class MLP:

    """
    MLP class:

    Attributes
    -----------

    n_layers: number of hidden layers
    layers: list of layers

    (questi boh vediamo)

    batch_size = 1;
    optimization_algorithm = gradient_descent()
    cost_function = MSE()
    regularization_function = NoReg()

    Methods
    --------

    add_layer(new_layer): adds new layer to layers
    predict(X): computes y_pred
    fit(X,y_true): trains MLP on the given training set


    """

    def __init__(self, n_layers, layer_units):

        """
        Build MLP 

        Parameters
        -----------
        n_layers : number of layers
        layer_units : list containing the number of units for each layer

        """

        self.n_layers = 0
        self.layers = []
        for l in range(n_layers):
            new_layer = Dense(layer_units[l-1], layer_units[l])
            self.add_layer(new_layer)

        self.batch_size = 1;
        self.optimization_algorithm = gradient_descent()
        self.cost_function = MSE()
        self.regularization_function = NoReg()


    def add_layer(self, new_layer):

        """
        Add new layer to layers

        Parameters
        -----------
        new_layer

        """
        self.layers.append(new_layer)
        self.n_layers += 1

    def predict(self, X):

        """
        Computes the predicted outputs of the MLP 

        Parameters
        -----------
        X : input matrix (n_samples x n_features)

        Returns
        -------
        y_pred : output vector (n_samples)

        """

        n_samples = X.shape[0]
        y_pred = np.empty(n_samples)
        for sample in range(n_samples):
            layer_output = X[sample]
            for layer in self.layers:
                layer_input = layer_output
                layer_output = layer.forward_propagation(layer_input)
            y_pred[sample] = layer_output
        return y_pred

    
    def fit(self, X, y_true):

        """
        (TO BE COMPLETED)

        Fits the weigths of the MLP 

        Parameters
        -----------
        X : input matrix (n_samples x n_features)
        y_true : target outputs (n_samples)

        """


        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/self.batch_size)
        batch = 0
        
        for batch in range(n_batches):
            y_pred = self.predict(X[batch * self.batch_size : (batch+1)*self.batch_size])
            cost = self.cost_function(y_pred, y_true[batch * self.batch_size : (batch+1)*self.batch_size])
            for layer in self.layers:
                grad_input = layer.backpropagation()


        
        return        
        
    

