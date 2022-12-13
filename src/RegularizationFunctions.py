import numpy as np

class RegularizationFunction():
    '''
    Base class for regularization functions

    Methods to override:
        __init__(self,alpha=0.1): Initialises the class with the apha parameter not implemented
            Input: Float
        __call__(self,w): Output of function not implemented
            Input: np.array
            Output: Error
        derivative(self,w): Derivative of function not implemented
            Input: np.array
            Output: Error
    '''
    def __init__(self, alpha=0.1):
        raise NotImplementedError
    def __call__(self, w):
        raise NotImplementedError
    def derivative(self, w):
        raise NotImplementedError


class NoReg(RegularizationFunction):
    '''
    Computes the Empy regularization effect

    Methods:
        __init__(self,alpha=0.1): Initialises the class with the apha parameter to not raise errors 
            Input: Float
        __call__(self,w): Output of function always 0
            Input: np.array
            Output: Float
        derivative(self,w): Derivative of function always 0
            Input: np.array
            Output: np.array
    '''
    def __init__(self, alpha=0.1):
        self.alpha = alpha
    
    def __call__(self, w):
        return 0
    
    def derivative(self, w):
        return np.zeros(w.shape)


class L1Reg(RegularizationFunction):
    '''
    Computes the L1 (Lasso) regularization effect, which is the absolute sum of weights in the model

    Methods:
        __init__(self,alpha=0.1): Initialises the class with the apha parameter, default 0.1
            Input: Float
        __call__(self,w): Output of function
            Input: np.array
            Output: Float
        derivative(self,w): Derivative of function
            Input: np.array
            Output: np.array
        
    '''
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))

    def derivative(self, w):
        return self.alpha * np.sign(w)

class L2Reg(RegularizationFunction):
    '''
    Computes the L2 (Ridge) regularization effect, which is the squared sum of weights in the model

    Methods:
        __init__(self,alpha=0.1): Initialises the class with the apha parameter, default 0.1
            Input: Float
        __call__(self,w): Output of function
            Input: np.array
            Output: Float
        derivative(self,w): Derivative of function
            Input: np.array
            Output: np.array
    '''
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.sum(np.square(w))

    def derivative(self, w):
        return 2 * self.alpha * w

class ElasticReg(RegularizationFunction):
    '''
    Computes the ElasticNet regularization effect, which is a sum of L1 and L2

    Methods:
        __init__(self,alpha=0.1): Initialises the class with the L1 and L2 apha parameter, default 0.1,0.1
            Input: 2 separated Floats
        __call__(self,w): Output of function
            Input: np.array
            Output: Float
        derivative(self,w): Derivative of function
            Input: np.array
            Output: np.array
        
    '''
    def __init__(self, alpha_l1=0.1, alpha_l2=0.1):
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

    def __call__(self, w):
        return self.alpha_l1 * np.sum(np.abs(w)) + self.alpha_l2 * np.sum(np.square(w))

    def derivative(self, w):
        return self.alpha_l1 * np.sign(w) + 2 * self.alpha_l2 * w

def get_regularization_instance(reg_type, alpha=0.1,beta=0.1):
    '''
    Returns the activation function indicated in the input if present

    Input: String and float
        reg_type: String rapresenting the name of the regularization function
        alpha: Float rapresenting the regularization parameter
    Output: RegularizationFunction
    '''
    if reg_type in ['L1', 'Lasso', 'lasso', 'l1']:
        return L1Reg(alpha)
    elif reg_type in ['L2', 'Ridge', 'ridge', 'l2']:
        return L2Reg(alpha)
    elif reg_type in ['Elastic', 'ElasticNet', 'elastic', 'elasticnet']:
        return ElasticReg(alpha,beta)
    elif reg_type in ['None', 'No', 'no', 'none']:
        return NoReg()
    else:
        raise ValueError('Regularization function not recognized')