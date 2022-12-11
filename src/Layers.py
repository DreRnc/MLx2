import numpy as np
#from Functions import #_inseriscisuperclasse_
from Functions import get_activation_instance



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
    


class Fully_Connected_Layer(Layer):

    """
    A Fully Connected layer is a collection of neurons that are fully connected to the previous layer.


    W_ij is the weight of unit i for input j.
    self._weights is a marix with dimensions (n_units x n_units_per_input)
    self._biases is a vector with dimension n_units
    
    """
    
    def __init__(self, n_units, n_inputs_per_unit, weights_scale = 0.01):

        """
        Initialize a fully connected layer.
        
        Parameters
        ----------
        n_units (int): number of units in the layer
        n_inputs_per_unit (int): number of inputs per unit (units in layer before)
        weights_scale (int): scale for initialization of weights


        Updates Needed
        --------------
        Must add randomization for weights

        """

        self.n_units = n_units
        self.n_input_per_unit = n_inputs_per_unit
        self._weights = np.ones((n_units, n_inputs_per_unit)) * weights_scale
        self._biases = np.ones(n_units) * weights_scale

    def get_params(self):

        """
        Gets the parameters from the layer.

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

        Parameters
        ----------
        params (dict): the values to set for the parameters of the layer.
            "weights" is a matrix
            "bias" is a vector

        """

        self._weights = params["weights"]
        self._biases = params["biases"]

    def forwardprop(self, input):
        """
        Perform linear transformation to input

        Parameters
        ----------
        input (vect) inputs for forward propagation from previous layer

        Returns
        -------
        Vector of outputs (one for each unit).

        """

        if np.shape(self._biases) != self.n_units:
            raise Exception("Dimension Error!")
        return np.matmul(self._weights, input) + self._biases

    def backprop(self):
        pass







class Activation_Layer(Layer):

    """
    An activation layer applies an activation function to the output of a layer

    Updates Needed
    --------------
    Checks on correct activation functions, implementation of aliases
    """

    def __init__(self, n_units, activation = "ReLU"):
        self.activation = get_activation_instance(activation)
        self.n_units = n_units
    
    def forwardprop(self, input):
        """
        Applies activation function to input.

        Parameters
        ----------
        input (vect) inputs for forward propagation from previous layer

        Returns
        -------
        Vector of outputs (one for each unit).

        """
        return self.activation(input)

    def backprop(self):
        pass



class Dense(Layer):

    """

    A Dense layer is a fully connected layer with an activation function

    """
