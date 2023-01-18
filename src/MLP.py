import numpy as np
import math
from sklearn.model_selection import train_test_split

from src.Layers import Layer, FullyConnectedLayer, Dense
from src.MetricFunctions import get_metric_instance, MetricFunction
from src.EarlyStopping import EarlyStopping

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

    def __init__(self, hidden_layer_units, input_size, output_size, activation_function = 'sigm', task = 'regression'):

        """
        Build MLP for regression (the last layer is fully connected)

        Parameters
        -----------
        hidden_layer_units (list) : number of units for each layer
        input_size (int)
        output_size (int)
        activation_function (str)
        task (str)

        """
        self.layers = []
        self.input_size = input_size
        self.output_size = output_size
        self.task = task

        layer_units = [input_size] + hidden_layer_units + [output_size]
        
        n_layers = len(layer_units) - 1 

        for l in range(1, n_layers +1):


            if l < n_layers:
                new_layer = Dense(layer_units[l], layer_units[l-1], activation_function)
            elif self.task == 'classification': 
                new_layer = Dense(layer_units[l], layer_units[l-1], "sigmoid")
            else:
                new_layer = FullyConnectedLayer(layer_units[l], layer_units[l-1])
                
            
            self.layers.append(new_layer)

    def evaluate_model(self, X, y_true, metric = 'generic'):
        """

        Evaluates model on a set, given a certain metric.

        Parameters
        -----------
        X (n_samples x n_inputs) : 
        y_true (n_samples x n_output) : 
        metric (str) : metric to evaluate

        """
        if metric != 'generic':
            eval_metric = metric
            eval_metric = get_metric_instance(eval_metric)
        elif type(metric) != str:
            if issubclass(MetricFunction, self._eval_metric):
                eval_metric = self._eval_metric
        else:
            if self.task == "classification":
                eval_metric = get_metric_instance('acc')
            else:
                eval_metric = get_metric_instance('mse')

        y_pred = self.predict(X)

        return eval_metric(y_true, y_pred)

    def fit(self, X, y_true, n_epochs, batch_size, X_test = None, y_test = None, error = "MSE", eval_metric = "default", regularization = "no", \
        alpha_l1 = 0, alpha_l2 = 0, weights_initialization = "scaled", weights_scale = 0.1, weights_mean = 0, step = 0.1, momentum = 0, Nesterov = False, backprop_variant = 'no', \
        early_stopping = True, validation_split_ratio = 0.1, verbose = False, patience = 10, tolerance = 0.01):

        """

        Fits the weigths and biases of the MLP 

        Parameters
        -----------
        X (n_samples x n_inputs) : 
        y_true (n_samples x n_output) : 

        n_epochs (int) :
        batch_size (int) :
        error (str) :
        regularization (str) :
        alpha_l1 (float) :
        alpha_l2 (float) : 
        initialization (str) : 
        weights_scale (float): 
        step (float) : 
        momentum (float) : 
        early_stopping (bool) :
        validation_split_ratio (float) :
        verbose (bool) :
        patience (int) :
        tolerance (float) :
        scale (float) :
        mean (float) : 
        """
        n_epochs = int(n_epochs)
        batch_size = int(batch_size)

        if eval_metric == "default":
            self._eval_metric = get_metric_instance(error)
        else:
            self._eval_metric = get_metric_instance(eval_metric)
        
        input_size = X.shape[1]
        try:
            output_size = y_true.shape[1]
        except:
            output_size = 1
        n_samples = X.shape[0]
        n_batches = math.ceil(n_samples/batch_size)
        self.learning_curve = np.zeros(n_epochs)
        self.validation_curve = np.zeros(n_epochs)
        self.learning_accuracy_curve = np.zeros(n_epochs)
        self.test_accuracy_curve = np.zeros(n_epochs)

        if input_size != self.input_size or output_size != self.output_size:
            raise Exception("Dimension Error!")


        for layer in self.layers:
            layer.initialize(weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, backprop_variant)

        error_function = get_metric_instance(error)

        # Initializes EarlyStopping
        if early_stopping:
            early_stopping = EarlyStopping(patience = patience, tolerance = tolerance, metric = self._eval_metric)
            early_stopping.initialize()
            X, X_test, y_true, y_test = train_test_split(X, y_true, test_size = validation_split_ratio, shuffle = True)
        
        # training_set = np.concatenate((X, y_true), axis = 1)
        n_batches = math.ceil(X.shape[0]/batch_size)
        
        # Training
        
        for epoch in range(n_epochs):

            # np.random.shuffle(training_set)
            
            # TR = np.split(training_set, [input_size], axis = 1)
            # X = TR[0]
            # y_true = TR[1]

            X_batches = np.split(X, range(batch_size, X.shape[0], batch_size), axis = 0)
            y_true_batches = np.split(y_true, range(batch_size, y_true.shape[0], batch_size), axis= 0)

            for batch in range(n_batches):

                X_batch = X_batches[batch]
                y_true_batch = y_true_batches[batch]

                y_pred_batch = self.predict(X_batch)
                
                grad_outputs = error_function.derivative(y_true_batch, y_pred_batch)
                
                for layer in reversed(self.layers):

                    grad_inputs = layer.backprop(grad_outputs)
                    grad_outputs = grad_inputs
            
            # Validation/Test Set: saving statistics and learning curve

            if X_test is not None:
                    y_pred_test = self.predict(X_test)
                    test_metric = self._eval_metric(y_test, y_pred_test)
                    self.validation_curve[epoch] = test_metric
                    if self.task == 'classification':
                        self.test_accuracy_curve[epoch] = get_metric_instance('acc')(y_test, y_pred_test)

            if early_stopping:

                params = [layer.get_params() for layer in self.layers]
                stop = early_stopping.on_epoch_end(test_metric, y_test, y_pred_test, params)

                if stop:
                    print(f"Early stopped training on epoch {epoch}")
                    best_params = early_stopping._best_params
                    for layer, layer_best_params in zip(self.layers, best_params):
                        layer.set_params(layer_best_params)
                    break

            y_pred = self.predict(X)

            self.learning_curve[epoch] = (self._eval_metric(y_true, y_pred))
            if self.task == 'classification':
                self.learning_accuracy_curve[epoch] = get_metric_instance('acc')(y_true, y_pred)

            if verbose:
                print("Epoch " + str(epoch) + ": " + "metric" + " = " + str(self._eval_metric(y_true, y_pred)))


    def predict(self, X):

        """
        Computes the predicted outputs of the MLP 

        Parameters
        -----------
        X : input matrix (n_samples x n_input)

        Returns
        -------
        y_pred : output matrix (n_samples x n_output)

        """
        
        (n_samples, input_size) = X.shape

        if input_size != self.input_size:
            raise Exception("Dimension Error!")

        y_pred = np.empty((n_samples, self.output_size))
        
        tmp = X
        for layer in self.layers:
            layer.input = tmp
            layer.output = layer.forwardprop(layer.input)
            tmp = layer.output

        y_pred = layer.output

        return y_pred