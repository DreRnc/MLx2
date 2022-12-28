from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from itertools import product
from src.MetricFunctions import get_metric_instance, ErrorFunction

class GridSearch():
    '''
    Class for Grid Search

    Attributes
    ----------
    model (Model): The model to be optimized
    parameters_grid (Dictionary): The combinations of parameters to be tested
    loss_function (MetricFunction): The loss function to be used
    n_results (Int): The number of results to be returned
    best_parameters (List): The best parameters found
    best_score (Float): The best score found
    best_model (Model): The best model found


    Methods
    -------
    fit(X, y, parameters_grid, n_folds = 0, stratified = False, test_size = 0.2, verbose = True):
        Performs the grid search
    get_best_parameters(n_results = 1, all = False): 
        Returns the best n parameters found with the scores
    '''


    def __init__(self, model, loss_function):
        '''
        Constructor

        Parameters
        ----------
        model (Model): The model to be optimized
        loss_function (MetricFunction): The loss function to be used
        n_results (Int): The number of results to be returned
        '''

        self.model = model

        if type(loss_function) == str:
            self.loss_function = get_metric_instance(loss_function)
        else:
            self.loss_function = loss_function
        

    def fit(self, X, y, parameters_grid, n_folds = 0, stratified = False, test_size = 0.2, verbose = True):
        
        '''
        Performs the grid search

        Parameters
        ----------
        X (np.array): The input data
        y (np.array): The output data
        parameters_grid (Dictionary): The values of parameters to be tested
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified
        verbose (Bool): If True prints the results of each combination of parameters
        '''
        self.data = X

        scores = []
        par = []

        self.parameters_grid = parameters_grid

        # if n_folds not int raise error
        if type(n_folds) != int:
            raise TypeError('n_folds must be an integer')

        if n_folds < 2:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = test_size, stratify = stratified)
        elif stratified: 
            folds = StratifiedKFold(n_splits = n_folds)
        else:
            folds = KFold(n_splits = n_folds)
        
        # Creates a list with all the combinations of parameters
        par_combinations = list(product(*list(self.parameters_grid.values())))

        for i, values in enumerate(par_combinations):
            parameters = {}

            for j, parameter in enumerate(self.parameters_grid.keys()):
                parameters[parameter] = values[j]

            if verbose:
                print(f'Combination {i+1}/{len(par_combinations)}')
                print(f'Parameters: {parameters}')

            if n_folds < 2:
                self.model.fit(X_train, y_train, **parameters)
                scores.append(self.loss_function(y_val, self.model.predict(X_val)))
            else:
                current_fold = 1
                score = 0
                for train_index, val_index in folds.split(X, y):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    self.model.fit(X_train, y_train, **parameters)
                    score += self.loss_function(y_val, self.model.predict(X_val))

                    if verbose:
                        print(f'Fold {current_fold}/{n_folds} score: {self.loss_function(y_val, self.model.predict(X_val))}')

                    current_fold += 1

                if verbose:
                    print(f'Validation score: {score/n_folds}')
                scores.append(score/n_folds)
                par.append(parameters)


        self.results = [ [scores[i], par[i]] for i in range(len(scores)) ]
        if isinstance(self.loss_function, ErrorFunction):
            self.results.sort(key = lambda x: x[0])
        else: 
            self.results.sort(key = lambda x: x[0], reverse = True)
        
        if verbose:
            print(f'Best parameters: {self.results[0][1]}')
            print(f'Best score: {self.results[0][0]}')
        self.best_parameters = self.results[0][1]
        self.best_score = self.results[0][0]

        self.model.fit(X, y, **self.best_parameters)
        self.best_model = self.model

    def get_best_parameters(self, n_parameters = 1, all = False):
        '''
        Returns the best n parameters

        Parameters
        ----------
        n_results (Int): The number of results to be returned
        all (Bool): If True returns all the results

        Returns
        -------
        List of dictionaries: The best n parameters
        '''
        
        if all:
            return self.results
        else:
            return self.results[:n_parameters]
    

        

















        
