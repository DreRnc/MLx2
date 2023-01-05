from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from itertools import product
from src.MetricFunctions import get_metric_instance, ErrorFunction
import random
import concurrent.futures
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

class GridSearch():
    '''
    Class for Grid Search

    Attributes
    ----------
    model (Model): The model to be optimized, after the fit method is called it contains the best model found
    parameters_grid (Dictionary): The combinations of parameters to be tested
    loss_function (MetricFunction): The loss function to be used in the optimization
    n_results (Int): The number of results to be returned
    best_parameters (List): The parameters of the best performing model
    best_score (Float): The best score achived
    best_model (Model): The best model found trained on the whole dataset provided


    Methods
    -------
    fit(X, y, n_folds = 1, test_size = 0.2, random_state = None, verbose = False, n_jobs = 1, **parameters_grid):
        Performs the grid search and saves the best parameters and the best model
    get_best_parameters(n_results = 1, all = False): 
        Returns the best n parameters found with the scores
    '''


    def __init__(self, model):
        '''
        Constructor

        Parameters
        ----------
        model (Model): The model to be optimized
        loss_function (MetricFunction): The loss function to be used in the optimization
        '''

        self.model = model

    def compute(self, values, l):

        '''
        This method is used to compute the score of a combination of parameters

        Parameters
        ----------
        values (List): The values of the parameters to be tested
        l (Int): The total number of combinations to be tested

        Returns
        -------
        Float: The score of the combination
        Dictionary: The parameters of the combination
        '''
    
        parameters = {}

        for j, parameter in enumerate(self.parameters_grid.keys()):
            parameters[parameter] = values[j]


        if self.n_folds < 2:
            self.model.fit(self.X_train, self.y_train, **parameters)
            score = self.model.evaluate_model(self.X_val, self.y_val)
        else:
            current_fold = 1
            score = 0
            for train_index, val_index in self.folds:
                X_train, X_val = self.X[train_index], self.X[val_index]
                y_train, y_val = self.y[train_index], self.y[val_index]

                self.model.fit(X_train, y_train, **parameters)

                score += self.model.evaluate_model(X_val, y_val)

                current_fold += 1

        self.i = self.i + 1

        if self.verbose:
            print('-----------------------------------')
            print(f'Combination {self.i}/{l}')        
            print(f'Parameters: {parameters}')
            print(f'Validation score: {score/self.n_folds}')
        
        return score / self.n_folds, parameters
        

    def create_folds(self, n_folds, stratified):
        '''
        Creates the folds for the cross validation

        Parameters
        ----------
        X (np.array): The input data
        y (np.array): The output data
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified

        Returns
        -------
        List: The folds
        '''

        # if n_folds not int raise error
        if type(n_folds) != int:
            raise TypeError('n_folds must be an integer')

        if n_folds < 2:
            if stratified:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size = self.test_size, stratify = self.y)
            else:
                self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size = self.test_size)

        elif stratified: 
            cv = StratifiedKFold(n_splits = n_folds)
            self.folds = list(cv.split(self.X, self.y))
        else:
            cv = KFold(n_splits = n_folds)
            self.folds = list(cv.split(self.X, self.y))

    def eta(self, par_combinations, get_eta):
        '''
        This method is used to compute the eta of the grid search

        Parameters
        ----------
        par_combinations (List): The combinations of parameters to be tested
        get_eta (Bool): If True the eta is computed
        '''

        if get_eta:
            print('Computing ETA')
            if len(par_combinations) < 100:
                pass
            else:
                # start counting the time
                start = time.time()
                eta_combinations = random.sample(par_combinations, 100)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.compute, values, len(eta_combinations)) for values in eta_combinations]

                    concurrent.futures.wait(futures)
                    #self.i = 0
                end = time.time()
                eta = (end - start) * len(par_combinations) / 100
                # get the time in hours
                eta = eta / 3600
                print(f'ETA: {eta} hours')

    def clean_output(self):
        '''
        This method is used to clean the output of the grid search
        '''

        self.results = [ [self.scores[i], self.par[i]] for i in range(len(self.scores)) ]

        if self.model.task == 'regression':
            self.results.sort(key = lambda x: x[0])
        elif self.model.task == 'classification':
            self.results.sort(key = lambda x: x[0], reverse = True)
        
        if self.verbose:
            print('\n')
            print(f'Best parameters: {self.results[0][1]}')
            print(f'Best score: {self.results[0][0]}')


        self.best_parameters = self.results[0][1]
        self.best_score = self.results[0][0]

        self.model.fit(self.X, self.y, **self.best_parameters)
        self.best_model = self.model

    def grid_search(self, par_combinations):
        '''
        This method is used to perform the grid search

        Parameters
        ----------
        par_combinations (List): The combinations of parameters to be tested
        '''

        if self.parallel:
            print('Parallelisation activated')

            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.compute, values, len(par_combinations)) for values in par_combinations]

                concurrent.futures.wait(futures)

                for future in concurrent.futures.as_completed(futures):
                    self.scores.append(future.result()[0])
                    self.par.append(future.result()[1])
        else:
            print('Parallelisation not active')
            for values in par_combinations:
                out = self.compute(values, len(par_combinations))
                self.scores.append(out[0])
                self.par.append(out[1])


    def fit(self, X, y, parameters_grid, n_folds = 1, stratified = False, test_size = 0.2, verbose = True, parallel = False, random_search = False, n_random = 10, get_eta = False):
        
        '''
        Performs the grid search

        Parameters
        ----------
        X (np.array): The input data
        y (np.array): The output data
        parameters_grid (Dictionary): The values of parameters to be tested
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified
        test_size (Float): The size of the test set if n_folds < 2
        verbose (Bool): If True prints the results of each combination of parameters
        parallel (Bool): If True uses all the cores of the CPU to compute the results
        random_search (Bool): If True takes n_random random combinations of parameters
        n_random (Int): The number of random combinations to be tested
        get_eta (Bool): If True returns the time it took to compute the results

        '''

        self.verbose = verbose
        self.n_folds = n_folds
        self.scores = []
        self.par = []
        self.X = X
        self.y = y
        self.parameters_grid = parameters_grid
        self.test_size = test_size
        self.parallel = parallel

        # Creates the folds
        self.create_folds(n_folds, stratified)
        
        # Creates a list with all the combinations of parameters
        #print(list(product(*list(self.parameters_grid.values()))))
        par_combinations = list(product(*list(self.parameters_grid.values())))

        #get the eta
        self.eta(par_combinations, get_eta)

        # If random search is True, it takes n_random random combinations
        if random_search:
            par_combinations = random.sample(par_combinations, n_random)
            if self.verbose:
                print(f'Random search of: {n_random} combinations')

        elif self.verbose:
            print(f'Grid search of combinations: {len(par_combinations)}')

        #self.i = 0

        self.grid_search(par_combinations)

        self.clean_output()



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



class RandomGridsearch(GridSearch):

    def random_sample(self, parameters_grid, n):
        """
        :param pbounds: a dictionary with the parameter names as keys and a tuple with minimum and maoutimum values.
        :param n: number of points to sample.
        """

        out = np.empty((n, len(parameters_grid)), dtype=object)
        # change out datatype to object
        out = out.astype(object)
        for i, (name, bounds) in enumerate(parameters_grid.items()):
            if type(bounds) == list:
                out[:, i] = np.random.choice(bounds, size=n)
            else:
                out[:, i] = np.random.uniform(bounds[0], bounds[1], size=n)
        return out

    def eta(self, par_combinations, get_eta):
        if get_eta:

            print('Getting ETA')

            if len(par_combinations) < 200:
                print('ETA: less than 1 hour')
            else:
                # start counting the time
                start = time.time()
                # get 100 random combinations
                eta_combinations = self.random_sample(self.parameters_grid, self.eta_len)
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.compute, values, len(eta_combinations)) for values in eta_combinations]

                    concurrent.futures.wait(futures)

                    for future in concurrent.futures.as_completed(futures):
                        self.scores.append(future.result()[0])
                        self.par.append(future.result()[1])

                end = time.time()
                eta = (end - start) * len(par_combinations) / len(eta_combinations)
                # get the time in hours
                eta = eta / 3600
                print(f'ETA: {eta} hours')
        


    def fit(self, X, y, parameters_grid, n_folds = 0, stratified = False, test_size = 0.2, verbose = True, parallel = False, n_random = 10, get_eta = False):
        
        
        '''
        Performs the grid search

        Parameters
        ----------
        X (np.array): The input data
        y (np.array): The output data
        parameters_grid (Dictionary): The values of parameters to be tested
        n_folds (Int > 1): The number of folds to be used in the cross validation
        stratified (Bool): If True the folds are stratified
        test_size (Float): The size of the test set if n_folds < 2
        verbose (Bool): If True prints the results of each combination of parameters
        parallel (Bool): If True uses all the cores of the CPU to compute the results
        random_search (Bool): If True takes n_random random combinations of parameters
        n_random (Int): The number of random combinations to be tested
        get_eta (Bool): If True returns the time it took to compute the results

        '''

        self.verbose = verbose
        self.n_folds = n_folds
        self.scores = []
        self.par = []
        self.X = X
        self.y = y
        self.parameters_grid = parameters_grid
        self.test_size = test_size
        self.parallel = parallel
        self.eta_len = 100
        # Creates the folds
        self.create_folds(n_folds, stratified)
        
        # Creates a list with all the combinations of parameters
        par_combinations = self.random_sample(parameters_grid, n_random)
        
        #get the eta
        self.eta(par_combinations, get_eta)

        # If random search is True, it takes n_random random combinations


        if self.verbose:
            print(f'Random search of combinations: {len(par_combinations)}')


        self.grid_search(par_combinations)

        self.clean_output()



        
