import numpy as np
from ActivationFunctions import get_activation_instance
from RegularizationFunctions import get_regularization_instance

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

    def initialize(self, optimizer, regularization_function, weights_initialization = 'scaled', weights_scale = 0.01):

        """
        
        Initialize properties of the fully connected layer which are specific for each fit.
        Function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (int): scale for initialization of weights
        optimizer (Optimizer): type of optimizer for layer
        regularization_function (RegularizationFunction): regularization function for the layer

        """

        # Weight initialization
        if weights_initialization == "scaled":
            scale = weights_scale
        elif self.weights_initialization == "xavier":
            scale = 1 / self.n_input
        elif self.weights_initialization == "he":
            scale = 2 / self.n_input
        self._weights = np.random.normal(loc = 0.0, scale = scale, size = (self.n_inputs_per_unit, self.n_units))
        self._biases = np.zeros(1, self.n_units)

        # Optimizer initialization
        self.optimizer = optimizer
        self.optimizer.initialize()

        # Regularization function
        self.regularization_function = regularization


    def get_params(self):

        """
        
        Gets the parameters from the layer.
        Function used for early stopping.

        Returns
        ----------
        Dictionary of parameters from the layer. 
            "weights" is a matrix
            "bias" is a vector

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

        if np.shape(self._biases)[0] != self.n_units:
            raise Exception("Dimension Error!")
        return np.matmul(self._weights, input) + self._biases # broadcasting

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

        grad_input = np.matmul(grad_output, self._weights.T)
        grad_weights = np.matmul(self._input, grad_output) + self.regularization.derivative(self._weights)
        grad_biases = grad_output.sum(axis = 0, keepdims = True) 

        weights_update, biases_update = self.optimizer.optimize(grad_weights, grad_biases)

        self._biases += biases_update
        self._weights += weights_update

        return grad_input




class ActivationLayer(Layer):

    """
    
    Activation layer applies an activation function elemnt-wise to the input

    """

    def __init__(self, n_units, activation = "ReLU"):

        """
        
        Initialize activation layer with its activation function and number of units.
        
        Parameters
        ----------
        n_units (int): number of units in the layer

        """

        self.n_units = n_units
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
        self._activation_layer = ActivationLayer(n_units, activation, activation)

    def initialize(self, optimizer, regularization, weights_initialization = 'scaled', weights_scale = 0.01):

        """
        
        Initialize properties of the FCL and AL which are specific for each fit.
        Function is infact called whenever starting a new fit.
        
        Parameters
        ----------
        weights_initialization (str): type of initialization for weights
        weights_scale (int): scale for initialization of weights
        optimizer (Optimizer): type of optimizer for layer
        regularization_function (RegularizationFunction): regularization function for the layer

        """

        self._fully_connected_layer.initialize(weights_initialization, optimizer, regularization)

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

