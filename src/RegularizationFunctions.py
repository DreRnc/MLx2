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
    def __init__(self):
        raise NotImplementedError
    def __call__(self, w):
        raise NotImplementedError
    def derivative(self, w):
        raise NotImplementedError

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
    def __init__(self, alpha_l1, alpha_l2):
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

    def __call__(self, w):
        return self.alpha_l1 * np.sum(np.abs(w)) + self.alpha_l2 * np.sum(np.square(w))

    def derivative(self, w):
        return self.alpha_l1 * np.sign(w) + 2 * self.alpha_l2 * w

class L1Reg(ElasticReg):
    '''
    Computes the L2 (Ridge) regularization effect, which is the squared sum of weights in the model

    '''
    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(alpha_l1, alpha_l2)
        self.alpha_l2 = 0

class L2Reg(ElasticReg):
    '''
    Computes the L2 (Ridge) regularization effect, which is the squared sum of weights in the model

    '''
    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(alpha_l1, alpha_l2)
        self.alpha_l1 = 0


class NoReg(ElasticReg):

    def __init__(self, alpha_l1, alpha_l2):
        super().__init__(alpha_l1, alpha_l2)
        self.alpha_l1 = 0
        self.alpha_l2 = 0


def get_regularization_instance(reg_type, alpha_l1, alpha_l2):
    '''
    Returns the activation function indicated in the input if present

    Input: String and float
        reg_type: String rapresenting the name of the regularization function
        alpha: Float rapresenting the regularization parameter
    Output: RegularizationFunction
    '''
    if reg_type in ['L1', 'Lasso', 'lasso', 'l1']:
        return L1Reg(alpha_l1, alpha_l2)
    elif reg_type in ['L2', 'Ridge', 'ridge', 'l2']:
        return L2Reg(alpha_l1, alpha_l2)
    elif reg_type in ['Elastic', 'ElasticNet', 'elastic', 'elasticnet']:
        return ElasticReg(alpha_l1, alpha_l2)
    elif reg_type in ['None', 'No', 'no', 'none']:
        return NoReg(alpha_l1, alpha_l2)
    else:
        raise ValueError('Regularization function not recognized')
