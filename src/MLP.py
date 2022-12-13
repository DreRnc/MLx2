import numpy as np
import math

from Layers import Layer, FullyConnectedLayer, Dense
from MetricFunctions import get_metric_instance

class MLP:

    """
    MLP class:

    Attributes
    -----------
    layers (list)
    input_size (int)
    output_size (int)

    Methods
    --------

    fit: fits MLP weigths on the given training set
    predict: computes y_pred

    """

    def __init__(self, hidden_layer_units, input_size, output_size, activation_function_str):

        """
        Build MLP 

        Parameters
        -----------
        hidden_layer_units (list) : number of units for each layer
        input_size (int)
        output_size (int)
        activation_function (str)

        """
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size

        layer_units = [input_size, hidden_layer_units, output_size]
        n_layers = len(layer_units) - 1;

        for l in range(n_layers):

            if l != n_layers -1:
                new_layer = Dense(layer_units[l], layer_units[l-1], activation_function_str)
            else:
                new_layer = FullyConnectedLayer(layer_units[l], layer_units[l-1])
            
            self.layers.append(new_layer)

    
    def fit(self, X, y_true, batch_size, inizialization_str, scale, error_function_str, optimizer_str, regularization_function_str):

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
        n_batches = math.ceil(n_samples/batch_size)

        for layer in self.layers:
            layer.initialize(inizialization_str, scale, optimizer_str, regularization_function_str)

        error_function = get_metric_instance(error_function_str)
        
        for batch in range(n_batches):

            if batch != n_batches - 1 :
                X_batch = X[batch * batch_size : (batch+1)*batch_size]
                y_true_batch = y_true[batch * batch_size : (batch+1)*batch_size]
            else:
                X_batch = X[batch * batch_size : -1]
                y_true_batch = y_true[batch * batch_size : -1]

            y_pred_batch = self.predict(X_batch)
            print(error_function_str + " = " + str(error_function(y_true_batch, y_pred_batch)))
            
            grad_outputs = error_function.derivative(y_true_batch, y_pred_batch)
            
            for layer in self.layers:

                grad_inputs = layer.backprop(grad_outputs)


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
        
        (n_samples, input_size) = X.shape
        if input_size != self.input_size:
            raise Exception("Dimension Error!")

        y_pred = np.empty((n_samples, self.output_size))
        for sample in range(n_samples):
            tmp = X[sample]
            for layer in self.layers:
                layer.input = tmp
                layer.output = layer.forwardprop(layer.input)
                tmp = layer.output
            y_pred[sample] = layer.output
        return y_pred

