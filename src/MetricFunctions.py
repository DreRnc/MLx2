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
    Methods:
        __call__(self,y_true, y_pred): Returns the accuracy
            Input:  2 np.arrays of the same shape
                y_true: np.array of the true values can be one hot encoded or binary
                y_pred: np.array of the predicted values can be one hot encoded or binary
            Output: Float

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
            Input: 2 np.arrays of the same shape
            Output: np.array

    '''
    def __call__(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        return np.mean(np.square(y_pred - y_true))

    def derivative(self, y_true, y_pred):
        if y_true.shape != y_pred.shape:
            raise ValueError("inputs must have the same shape")
        return 2 * (y_pred - y_true) / y_true.shape[0]

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
        return (y_pred - y_true) / np.mean(np.abs(y_pred - y_true)) / y_true.shape[0]


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
    else:
        raise ValueError("Metric function not found")




