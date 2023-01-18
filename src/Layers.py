import numpy as np
from src.ActivationFunctions import get_activation_instance
from src.RegularizationFunctions import get_regularization_instance
from src.Optimizers import HeavyBallGradient
from math import sqrt

class Layer:

    """
    
    A Layer is a collection of neurons

    Base Methods
    ------------
    __init__
    get_params
    update_params

    Override Methods
    ----------------
    forwardprop
    backprop
    
    """

    def __init__(self):
        
        pass

    def get_params(self):
        
        """
        
        Gets parameters
        
        """
        
        pass

    def update_params(self):
        
        """
        
        Update parameters after batch/online
        
        """
        
        pass

    def forwardprop(self):

        raise NotImplementedError

    def backprop(self):

        raise NotImplementedError
    


class FullyConnectedLayer(Layer):

    """
    
    A Fully Connected layer is a collection of neurons that are fully connected to the previous layer.


    W_ij is the weight of unit i for input j.
    self._weights is a marix with dimensions (n_inputs_per_unit x n_units)
    self._biases is a vector with dimension n_units
    
    """
    
    def __init__(self, n_units, n_inputs_per_unit):

        """
        
        Initialize only properties of the layer that are intrinsic to the structure of the MLP.
        
        Parameters
        ----------
        n_units (int): number of units in the layer
        n_inputs_per_unit (int): number of inputs per unit (units in layer before)

        """

        self.n_units = n_units
        self.n_inputs_per_unit = n_inputs_per_unit

    def initialize(self, weights_initialization, weights_scale, weights_mean, regularization_function, alpha_l1, alpha_l2, step, momentum, Nesterov, backprop_variant):

        """
        
        Initialize properties of the fully connected layer which are specific for each fit.
        Function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        
        weights_initialization (str): type of initialization for weights
        weights_scale (int): std of the normal distribution for initialization of weights
        weights_mean (int): mean of the normal distribution for initialization of weights
        regularization_function (RegularizationFunction): regularization function for the layer
        alpha_l1
        alpha_l2
        step
        momentum
        Nesterov

        """
        scale = weights_scale
        # Weight initialization
        if weights_initialization == "scaled":
            self._weights = np.random.normal(loc = weights_mean, scale = weights_scale, size = (self.n_inputs_per_unit, self.n_units))
        elif weights_initialization == "xavier":
            # 1 / sqrt(n_inputs_per_unit) with numpy
            bound = 1 / sqrt(self.n_inputs_per_unit)
            self._weights = np.random.uniform(low = -bound, high = bound, size = (self.n_inputs_per_unit, self.n_units))

        elif weights_initialization == "he":
            bound = 2 / sqrt(self.n_inputs_per_unit)
            self._weights = np.random.uniform(low = -bound, high = bound, size = (self.n_inputs_per_unit, self.n_units))
        else:
            print("invalid weigths initialization: choose one between 'scaled', 'xavier', 'he' ")

        
        self._biases = np.zeros((1, self.n_units))

        #save last weigths and biases update for HBG
        self._last_weights_update = 0
        self._last_biases_update = 0

        self._last_grad_weights = 0
        self._last_grad_biases = 0

        # Optimizer initialization
        self.optimizer = HeavyBallGradient(step, momentum, Nesterov)

        # Backpropagation variant: 'no', 'rprop', 'quickprop'
        self.backprop_variant = backprop_variant 

        # Regularization function
        self.regularization_function = get_regularization_instance(regularization_function, alpha_l1, alpha_l2)


    def get_params(self):

        """
        
        Gets the parameters from the layer.
        Function used for early stopping.

        Returns
        ----------
        Dictionary of parameters from the layer. 
            "weights" is a matrix
            "bias" is a vectormodel_search.get_best_parameters

        """

        return {"weights": self._weights.copy(), "biases": self._biases.copy()}

    def set_params(self, params):

        
        """
        
        Sets the parameters of the layer.
        Function used for early stopping.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.
            "weights" is a matrix
            "bias" is a horizontal vector

        """

        self._weights = params["weights"]
        self._biases = params["biases"]

    def forwardprop(self, input):
        
        """
        
        Perform linear transformation to input

        Parameters
        ----------
        Matrix of inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        Matrix of outputs of whole batch (batch_size x n_units)

        """
        self._input = input         # saves values for backprop

        if np.shape(self._biases)[1] != self.n_units:
            raise Exception("Dimension Error!")

        return np.matmul(input, self._weights) + self._biases # broadcasting



    def backprop(self, grad_output):
        

        """
       
        Performs backpropagation, updating weights and biases, and passing gradient for next step.
        Starts by calculating various gradients with respect to input, weights and biases.
        Then calls optimizer to update weitghs and biases.
        Finally, returns gradient with respect to input.

        Parameters
        ----------
        Gradient of loss function with respect to output of this layer

        Returns
        -------
        Gradient of loss function with respect to input of this layer (output of previous layer)

        """

        weights = self._weights
        biases = self._biases

        if self.optimizer.Nesterov:
            weights = weights + self.optimizer.momentum * self._last_weights_update
            biases = biases + self.optimizer.momentum * self._last_biases_update

        grad_input = np.matmul(grad_output, weights.T)
        grad_weights = np.matmul(self._input.T, grad_output) + self.regularization_function.derivative(weights)
        grad_biases = grad_output.sum(axis = 0, keepdims = True)

        weights_update, biases_update = self.optimizer(grad_weights, grad_biases, \
            self._last_weights_update, self._last_biases_update, self.backprop_variant, self._last_grad_weights, self._last_grad_biases)

        if self.backprop_variant == 'quickprop':
            self._last_grad_weights = grad_weights
            self._last_grad_biases = grad_biases

        self._biases += biases_update
        self._weights += weights_update

        self._last_weights_update = weights_update
        self._last_biases_update = biases_update

        return grad_input



class ActivationLayer(Layer):

    """
    
    Activation layer applies an activation function elemnt-wise to the input

    """

    def __init__(self, activation = "ReLU"):

        """
        
        Initialize activation layer with its activation function and number of units.
        
        Parameters
        ----------
        n_units (int): number of units in the layer

        """

        self.activation = get_activation_instance(activation)
    
    def forwardprop(self, input):
        
        """

        Applies activation function to input.

        Parameters
        ----------
        Matrix of inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        Matrix of outputs of whole batch (batch_size x n_units)

        """

        self._input = input         # saves values for backprop

        # print(self.activation(input))

        return self.activation(input)

    def backprop(self, grad_output):
        
        """
        
        Performs backpropagation, undoing activation function.

        Parameters
        ----------
        Gradient of loss function with respect to output of this layer

        Returns
        -------
        Gradient of loss function with respect to input of this layer

        """

        return grad_output * self.activation.derivative(self._input)



class Dense(Layer):

    """

    A Dense layer is a fully connected layer with an activation layer afterwards 

    """

    def __init__(self, n_units, n_inputs_per_unit, activation = "ReLU"):

        """
        
        Initialize only properties of the layer that are intrinsic to the structure of the MLP.
        
        Parameters
        ----------
        n_units (int): number of units in the layer
        n_inputs_per_unit (int): number of inputs per unit (units in layer before)

        """

        self._fully_connected_layer = FullyConnectedLayer(n_units, n_inputs_per_unit)
        self._activation_layer = ActivationLayer(activation)

    def initialize(self, weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, backprop_variant):

        """
        
        Initialize properties of the FCL and AL which are specific for each fit.
        Function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (int): scale for initialization of weights
        regularization_function (RegularizationFunction): regularization function for the layer
        alpha_l1
        alpha_l2
        step
        momentum
        Nesterov

        """

        self._fully_connected_layer.initialize(weights_initialization, weights_scale, weights_mean, regularization, alpha_l1, alpha_l2, step, momentum, Nesterov, backprop_variant)

    def get_params(self):

        """
        
        Gets the parameters from the FCL layer.
        Function used for early stopping.

        Returns
        ----------
        Dictionary of parameters from the layer. 
            "weights" is a matrix
            "bias" is a vector

        """

        return self._fully_connected_layer.get_params()

    def set_params(self, params):

        
        """
        
        Sets the parameters of the FCL layer.
        Function used for early stopping.

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.
            "weights" is a matrix
            "bias" is a horizontal vector

        """

        self._fully_connected_layer.set_params(params)

    def forwardprop(self, input):
        
        """

        Applies activation function to input.

        Parameters
        ----------
        Matrix of inputs of whole batch (batch_size x n_inputs_per_unit)

        Returns
        -------
        Matrix of outputs of whole batch (batch_size x n_units)

        """

        output_FCL = self._fully_connected_layer.forwardprop(input)
        return self._activation_layer.forwardprop(output_FCL)


    def backprop (self, grad_output):

        """
        
        Performs backpropagation, combining backprop of both FCL and AL.
        First calculates gradient with respect to output of FCL.
        Then updates weights and biases.
        Finally calculates gradient with respect to input and returns it.

        Parameters
        ----------
        Gradient of loss function with respect to output of this layer.

        Returns
        -------
        Gradient of loss function with respect to input of this layer.

        """

        grad_output_FCL = self._activation_layer.backprop(grad_output)

        return self._fully_connected_layer.backprop(grad_output_FCL)

