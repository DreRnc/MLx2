import numpy as np
from src.MetricFunctions import MetricFunction, get_metric_instance, Accuracy


class EarlyStopping():

    def __init__ (self,  patience, tolerance, metric = "loss"):

        """
        Initialize EarlyStopping Object
        
        Parameters
        ----------

        metric (str or MetricFunction, default = "loss") : we don't pass directly the value in case a different metric is wanted
        patience (float) :
        tolerance (float) :
        mode (str, default = "min") : can be both "min" and "max", "max" needed for accuracy (in general, positive metrics)
        """

        if type(metric) == str:
            if metric == "loss":
                self.metric = metric
            else:
                self.metric = get_metric_instance(metric)
        elif isinstance(metric, MetricFunction):
            self.metric = metric 
        else:
            raise ValueError("The metric must be a Metric object or as string alias for it, or 'loss'.")

        self.patience = patience
        self.tolerance = tolerance

        if isinstance(self.metric, Accuracy):
            self.mode = "max"
        else:
            self.mode = "min"

    def initialize (self):

        """
        Initializes parameters, called before every training cycles.

        Parameters
        ----------
        _best_metric_value (float) :
        _n_epochs (int) :
        _n_worsening_epochs (int) :

        """
        if self.mode == 'min':
            self._best_metric_value = np.infty 
        elif self.mode == 'max':
            self._best_metric_value = -np.infty
        self._n_epochs = 0
        self._n_worsening_epochs = 0

    def on_epoch_end(self, loss, y_true, y_pred, params):

        """
        At the end of every epoch, evaluates situation, ran on validation set.

        Parameters
        ----------
        loss (float): the loss at the current epoch of training for the target dataset
        y_true (np.ndarray, shape: (n_samples, n_output_features)): the ground truth values for the validation set
        y_pred (np.ndarray, shape: (n_samples, n_output_features)): the predicted values for the validation set
        params (list): the current parameters of the model.

        Returns (bool): True if training has to stop, False otherwise.

        """

        self._n_epochs += 1

        if self.metric == "loss":
            metric_value = loss
        else:
            metric_value = self.metric(y_true, y_pred)

        if (self.mode == "min" and metric_value < self._best_metric_value - self.tolerance) or (self.mode == "max" and metric_value > self._best_metric_value + self.tolerance):
            self._best_metric_value = metric_value
            self._n_worsening_epochs = 0
            self._best_params = params
            self._best_epoch = self._n_epochs
        else:
            self._n_worsening_epochs += 1
            if self._n_worsening_epochs == self.patience:
                return True
        
        return False