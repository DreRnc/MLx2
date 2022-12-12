import numpy as np
import math

from Layers import Layer, Fully_Connected_Layer, Dense
from MetricFunctions import GetMetricFunction
from RegularizationFunctions import GetRegularizationFunction
from ActivationFunctions import GetActivationFunction

class MLP:

    """
    MLP class:

    Attributes
    -----------
    layers: list of layers

    Methods
    --------

    add_layer: adds new layer to layers
    predict: computes y_pred
    fit: fits MLP weigths on the given training set

    """

    def __init__(self, n_hidden_layers, hidden_layer_units, input_size, output_size, activation_function="Sigm"):

        """
        Build MLP 

        Parameters
        -----------
        n_layers : number of layers
        layer_units : list containing the number of units for each layer

        """
        self.layers = []

        layer_units = [input_size, hidden_layer_units, output_size]

        for l in range(n_hidden_layers):
            new_layer = Dense(layer_units[l], layer_units[l-1], activation_function)
            self.add_layer(new_layer)
        
        output_layer = Fully_Connected_Layer(output_size, input_size)

        self.add_layer(output_layer)


    def add_layer(self, new_layer):

        """
        Add new layer to layers
        
        (Forse per ora messa così è inutile?)

        Parameters
        -----------
        new_layer

        """
        self.layers.append(new_layer)

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
            tmp = X[sample]
            for layer in self.layers:
                layer.input = tmp
                layer.output = layer.forward_propagation(layer_input)
                tmp = layer.output
            y_pred[sample] = layer.output
        return y_pred

    
    def fit(self, X, y_true, batch_size, error_function_str, regularization_function_str):

        """

        Fits the weigths and biases of the MLP 

        Parameters
        -----------
        X : input matrix (n_samples x n_features)
        y_true : target outputs (n_samples)

        batch_size (int) 
        regularization_function (str)
        cost_function (str)

        """

        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/self.batch_size)

        error_function = GetMetricFunction(error_function_str)
        #regularization_function = GetRegularizationFunction(regularization_function_str)
        
        for batch in n_batches:

            if batch != n_batches - 1 :
                X_batch = X[batch * batch_size : (batch+1)*batch_size]
                y_true_batch = y_true[b * batch_size : (batch+1)*batch_size]
            else:
                X_batch = X[batch * batch_size : -1]
                y_true_batch = y_true[batch * batch_size : -1]

            y_pred_batch = self.predict(X_batch)
            grad_outputs = error_function.derivative(y_true_batch, y_pred_batch)

            for layer in self.layers:

                grad_inputs = layer.backprop(grad_outputs)


        
    

