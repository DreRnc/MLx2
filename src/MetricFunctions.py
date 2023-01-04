import numpy as np

class MetricFunction():
    '''
    Base class for metric functions

    Methods to override:
        __call__(self,y_true, y_pred): Returns the metric not implemented
            Input: np.array
            Output: Error
    '''
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

class Accuracy(MetricFunction):
    '''
    Computes the accuracy between two np.arrays of all sizes
    
    Atributes:
    ----------
    y_true: np.array of the true values
    y_pred: np.array of the predicted values

    Methods:
    --------
    __call__(self,y_true, y_pred): Returns the accuracy
        Input: np.array
            y_true: np.array of the true values
            y_pred: np.array of the predicted values
        Output: Accuracy
        

    '''
    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape: 
            raise ValueError("inputs must have the same shape")
        if y_true.shape[1]>1: #for the case of one hot encoding
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else: #for the case of binary classification
            y_pred = np.round(y_pred)
        return np.mean(y_true == y_pred)

class ErrorFunction(MetricFunction):
    '''
    Base class for error functions

    Methods to override:
        __call__(self,y_true, y_pred): Returns the error not implemented
            Input: np.array
                y_true: np.array of the true values
                y_pred: np.array of the predicted values 
            Output: Error
        derivative(self,y_true, y_pred): Returns the derivative of the error not implemented
            Input: 2 np.array of the same shape
            Output: Error
    '''
    
    def derivative(self, y_true, y_pred):
        raise NotImplementedError

class MSE(ErrorFunction):
    
    '''
    Computes the mean squared error between two np.arrays of all sizes 
    IMPORTANT TO BE DONE MAYBE WE SHOULD DEVIDE THE ERROR BY 2 TO SIMPLIFY BACK PRO.
    Methods:
        __call__(self,y_true, y_pred): Returns the mean squared error
            Input: 2 np.arrays of the same shape In the case of more than one output the array must have the shape (n_samples, n_outputs)
                y_true: np.array of the true values
                y_pred: np.array of the predicted values
            Output: Float
        derivative(self,y_true, y_pred): Returns the derivative of the mean squared error
            Input: 2 np.arrays (n_observations, n_outputs) of the same shape
            Output: np.array (n_observations, n_outputs)

    '''
    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        return np.mean(np.square(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        #we are deviding by the number of samples since we are using the mean of th gradients in a batch and not the sum
        # we are also not multiplying by 2 to simplify the backpropagation
        return (y_pred - y_true) / y_true.shape[0]

class MAE(ErrorFunction):
    '''
    Computes the mean absolute error between two np.arrays of all sizes
    Methods:
        __call__(self,y_true, y_pred): Returns the mean absolute error
            Input: 2 np.arrays of the same shape. In the case of more than one output the array must have the shape (n_samples, n_outputs)
                y_true: np.array of the true values
                y_pred: np.array of the predicted values
            Output: Float
        derivative(self,y_true, y_pred): Returns the derivative of the mean absolute error
            Input: 2 np.arrays of the same shape
            Output: np.array

    '''
    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        return np.mean(np.abs(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        return np.sign(y_pred-y_true)/(y_true.shape[0]*y_true.shape[1])


class NLL(ErrorFunction):
    '''
    Computes the negative log likelihood between two np.arrays of all sizes.

    Methods:
        __call__(self,y_true, y_pred): Returns the mean absolute error
            Input: 2 np.arrays of the same shape. In the case of more than one output the array must have the shape (n_samples, n_classes)
                y_true: np.array of the true class (one hot vector with 1 in the right class)
                y_pred: np.array of the predicted values (softmax distribution over the classes)
            Output: Float
        derivative(self,y_true, y_pred): Returns the derivative of negative log likelihood
            Input: 2 np.arrays of the same shape
            Output: np.array

    '''

    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
            
        # Calculate the negative log likelihood
        loss = -np.sum(y_true * np.log(y_pred))

        # Average the loss across the batch
        return loss / y_true.shape[0]

    def derivative(self, y_true, y_pred):

        # This is actually directly with respect to the weights.
        
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")

        # Average the gradient across the batch
        return (y_pred - y_true) / y_true.shape[0]


def get_metric_instance(metric):
    '''
    Returns the metric function indicated in the input if present
    Input: String
        metric: String rapresenting the name of the metric function
    Output: MetricFunction
    '''
    if metric in  ["MSE", "mean_squared_error",'mse','mean squared error']:
        return MSE()	
    elif metric in ["MAE", "mean_absolute_error",'mae','mean absolute error','mee','MEE','mean expected error','Mean Expected Error']:
        return MAE()
    elif metric in ["Accuracy", "accuracy", "acc", "ACC", "ACCURACY",'a']:
        return Accuracy()
    elif metric in ["Negative Log Likelihodd", "NLL"]:
        return NLL()
    else:
        raise ValueError("Metric function not found")




