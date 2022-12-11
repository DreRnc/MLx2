import numpy as np
import math

from Layers import Layer, Fully_Connected_Layer, Dense

from Functions import get_activation_instance

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

        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/self.batch_size)
        batches = [{"X" : X[b * batch_size : (b+1)*batch_size], \
            "y_true" :  y_true[b * batch_size : (b+1)*batch_size]} \
            for b in range(n_batches)]
        batches.append({"X" : X[(n_batches-1) * batch_size : -1], "y_true" : y_true[(n_batches-1) * batch_size : -1]})
        
        for batch in batches:

            for layer in self.layers:
                layer.get_params["weigths"]

                grad_weigths = layer.backprop(batch["X"], batch["y"])

                new_weigths = layer.weigths - optimization_step * grad_weigths
                layer.set_params({"weigths" : new_weigths})

        
    

