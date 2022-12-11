import numpy as np
import math

from Layers import Layer, FullyConnected, Dense

from Functions import get_activation_instance, get_regularization_instance, get_error_function

class MLP:

    """
    MLP class:

    Attributes
    -----------

    n_layers: number of hidden layers
    layers: list of layers

    Methods
    --------

    add_layer: adds new layer to layers
    predict: computes y_pred
    fit: fits MLP weigths on the given training set

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

    
    def fit(self, X, y_true, batch_size, optimization_step, regularization_function_str, error_function_str):

        """
        (TO BE COMPLETED)

        Fits the weigths of the MLP 

        Parameters
        -----------
        X : input matrix (n_samples x n_features)
        y_true : target outputs (n_samples)

        batch_size : 
        optimization_algorithm :
        regularization_function : 
        cost_function :

        """


        regularization_function = get_regularization_instance(regularization_function_str)
        error_function = get_error_function(error_function_str)

        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/self.batch_size)
        batches = [{"X" : X[b * batch_size : (b+1)*batch_size], \
            "y_true" :  y_true[b * batch_size : (b+1)*batch_size]} \
            for b in range(n_batches - 1)]
        batches.append({"X" : X[(n_batches-1) * batch_size : -1], "y_true" : y_true[(n_batches-1) * batch_size : -1]})
        
        for batch in batches:
            
            for layer in self.layers:
                layer.get_params["weigths"]
                grad_weigths = layer.backprop(X[sample], y[sample])
                new_weigths = optimization_algorithm(weigths, optimization_step, grad_weigths)
                layer.set_params({"weigths" : new_weigths})

        
        return        
        
    

