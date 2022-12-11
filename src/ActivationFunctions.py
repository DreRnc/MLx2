import numpy as np

class ActivationFunction():
    '''
    Base class for activation functions

    Methods:
        __call__(self,x): Output of function not implemented
            Input: np.array
            Output: Error
        derivative(self,x): Derivative of function not implemented
            Input: np.array
            Output: Error
    '''
    def __call__(self, x):

        raise NotImplementedError

    def derivative(self, x):

        raise NotImplementedError


class Identity(ActivationFunction):
    '''
    Identity Activation Function returns the input it gets

    Methods:
        __call__(self,x): Output of function
            Input: np.array
            Output: np.array
        derivative(self,x): Derivative of function
            Input: np.array
            Output: np.array
    '''
    def __call__(self, x):
        return x
    def derivative(self, x):
        return np.ones(x.shape)


class Sigmoid(ActivationFunction):
    '''
    Computes Sigmoid Activation Function

    Methods:
        __call__(self,x): Output of function
            Input: np.array
            Output: np.array
        derivative(self,x): Derivative of function
            Input: np.array
            Output: np.array
    '''
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    def derivative(self, x):
        return self(x) * (1 - self(x))


class Tanh(ActivationFunction):
    '''
    Computes HyperbolicThangent Activation Function

    Methods:
        __call__(self,x): Output of function
            Input: np.array
            Output: np.array
        derivative(self,x): Derivative of function
            Input: np.array
            Output: np.array
    '''
    def __call__(self, x):
        return np.than(x)
    
    def derivative(self, x):
        return 1-np.square(np.than(x))
    

class ReLU(ActivationFunction):
    '''
    Computes ReLU Activation Function

    Methods:
        __call__(self,x): Output of function
            Input: np.array
            Output: np.array
        derivative(self,x): Derivative of function
            Input: np.array
            Output: np.array
    '''
    def __call__(self, x):
        return np.maximum(0, x)
        
    def derivative(self, x):
        return (x > 0).astype(int)


def GetActivationFunction(activation):
    '''
    Returns the activation function indicated in the input if present
    Input: String
        activation: String rapresenting the name of the activation function
    Output: ActivationFunction
    '''
    if activation in ['sigmoid', 'Sigmoid', 'Sigmoid()','sig']:
        return Sigmoid()
    elif activation in ['tanh', 'Tanh', 'Tanh()','tanh','ta'] :
        return Tanh()
    elif activation in ['identity', 'Identity', 'Identity()','id']:
        return Identity()
    elif activation in ['relu', 'ReLU', 'ReLU()','r','RELU','Relu','re']:
        return ReLU()
    else:
        raise ValueError("Activation function not found")