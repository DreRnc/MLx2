import numpy as np
import math

from Layers.py import layer, 

class MLP:
    def __init__(self, n_layers, units_per_layer):
        self.n_layers = 0
        self.layers = []
        for l in range(n_layers):
            self.add_layer(units_per_layer[l])

    def add_layer(self, n_units):
        new_layer = Dense(n_units)
        self.layers.append(new_layer)
        self.n_layers += 1

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.empty(n_samples)
        for sample in range(n_samples):
            layer_output = X[sample]
            for layer in self.layers:
                layer_input = layer_output
                layer_output = layer.forward_propagation(layer_input)
            y_pred[sample] = layer_output
        return y_pred

    
    def fit(self, X, y_true, batch_size, optimization_algorithm):
        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/batch_size)
        batch = 0
        
        for batch in range(n_batches):
            y_pred = self.predict(X[batch * batch_size : (batch+1)*batch_size])
            
            for layer in self.layers:
                grad_input = layer.backpropagation



        
        return        
        
    

