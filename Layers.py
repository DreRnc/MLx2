import numpy as np

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
    


class Fully_Connected(Layer):

    """
    A Fully Connected layer is a collection of neurons that are fully connected to the previous layer.

    """
    def __init__(self, n_units, n_inputs_per_unit, weights_scale = 0.01):
        """
        Initialize a fully connected layer.
        
        Parameters
        ----------
        n_units (int): number of units in the layer
        n_inputs_per_unit (int): number of inputs per unit (units in layer before)
        weights_scale (int): scale for initialization of weights

        """

        self.n_units = n_units
        self.n_input_per_unit = n_inputs_per_unit
        self.weights_scale = weights_scale

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

    def forwardprop(self, X):
        """
        Perform linear transformation to input

        Parameters
        ----------
        X (vect)

        Returns
        -------
        Vector of outputs (one for each unit).
        """
        if np.shape(self._biases) != self.n_units:
            raise "Dimension Error!"
        return np.matmul(self._weights,X) + self._biases

    def backprop(self,)



class Activation(Layer):

    """
    An activation layer applies an activation function to the output of a layer

    """



class Dense(Layer):

    """
    A Dense layer is a fully connected layer with an activation function

    """