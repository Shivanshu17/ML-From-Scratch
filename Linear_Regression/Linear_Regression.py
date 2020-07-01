# This will be the implementation of Linear Regression from Scatch 
import sys
sys.path.append('/home/epoch/Shivanshu/Git-Projects/ML-From-Scratch/optim')
# The following files will be imported from the path appended above
import loss_functions as lf
import sgd
import Adagrad
import Adadelta
import Adam

import pandas as pd
import numpy as np

class LinearRegression():
    def __init__(self, data, epoch = 20, lr = 0.8, cost = 0, basis = 0, polynomial_basis_order = 1, optimization = 0, ep = 1e-8, q = 0.7, alpha = 0.9, b1 = 0.9, b2 = 0.99, h_p = 0.9):
        '''
        This class object will fit a linear regression model to the given data according to the prescribed basis & regularization
        We are gonna assume that the data being fed has been normalized already, and I am not gonna implement regularization for now
        Args:
            data (DataFrame) -> Original dataframe; this might be adjusted during the set_basis stage. Also, it has the output values in the last column.
            epochs (int) -> Defines the number of epochs the user wants to train this data for.
            lr (float) -> Determines the learning rate of the algorithm
            q (float) -> Represents the quantile loss function for when quantile cost function is used.
            h_p (float) -> Huber point for huber loss function
            cost (int) -> Defines the loss function as follows:
                0 - Mean Squared Error
                1 - Mean Absolute Error
                2 - Huber Loss
                3 - Log Cosh loss function
                4 - Quantile Error
            
            basis (int) -> Defines the basis to be used as follows:
                0 - Linear Basis
                1 - Polynomial Basis
                2 - Gaussian Basis
            
            optimization (int) -> Define the optimization function to be used:
                0 - SGD
                1 - Adagrad
                2 - Adadelta
                3 - Adam
            
        Returns:
            predictions (iterable) -> predict function returns the set of predictions for the trained model
            params (iterable) -> get_params() function returns the parameters (either before or after the training)
            score (float) -> Defines the loss score on the test set
            
        
        '''
        # Notice how we didn't take activation function as input, because linear regression implicitly performs linear activation.
        self.data  = data
        self.epoch = epoch
        self.lr = lr
        self.cost = cost
        self.basis = basis
        self.optimization = optimization
        self.polynomial_basis_order = polynomial_basis_order
        self.set_basis()
        self.ep = ep
        self.alpha = alpha
        self.b1 = b1
        self.b2 = b2
        self.h_p = h_p
        self.q = q
            
    def set_basis(self):
        '''
        This function sets the basis of the linear regression model and updates the data accordingly.
        
        
        '''
        if self.basis == 0:
            self.mapped_data = self.data.copy()
        if self.basis == 1:
            if self.polynomial_basis_order <= 1:
                raise ValueError("Order of polynomial basis should be more than one")
            output_frame = self.data.iloc[:, -1]
            mapped_array = []
            num_instance = len(self.data)
            for i in range(num_instance):
                list_to_append = []
                data_row = self.data.iloc[i,:]
                for j in range(len(data_row.columns)-1):
                    data_element = self.data.iloc[i,j]
                    for k in range(1, self.polynomial_basis_order):
                        list_to_append.append(data_element**k)
                mapped_array.append(list_to_append)
            self.mapped_data = pd.DataFrame(mapped_array)
            self.mapped_data = pd.concat([self.mapped_data, output_frame], axis = 1)
        #if self.basis == 2:
            #Still have to figure out this part
            
        
    def set_test_basis(self, data, basis, order = 1):
        '''
        This function sets the basis of the linear regression model and updates the data accordingly.
        Args:
            data (dataframe) -> contains the test data
            basis (int) -> Defines the basis of funciton
            order (int) -> Definest the polynomial order, if applicable
        
        
        '''
        if basis == 0:
            mapped_data = data.copy()
        if basis == 1:
            if order <= 1:
                raise ValueError("Order of polynomial basis should be more than one")
            mapped_array = []
            num_instance = len(data)
            for i in range(num_instance):
                list_to_append = []
                data_row = data.iloc[i,:]
                for j in range(len(data_row.columns)):
                    data_element = data.iloc[i,j]
                    for k in range(1, order):
                        list_to_append.append(data_element**k)
                mapped_array.append(list_to_append)
            mapped_data = pd.DataFrame(mapped_array)
        #if basis == 2:
            #Still have to figure out this part
            
        
        return mapped_data
    
        
    def set_params(self):
        '''
        This function intialises the parameters based on the basis and data. It is called by fit function.
        We are gonna assume that we are trying to train for a convex function, and therefore we can safely initialize 
        the parameters at 0.
        Returns:
            params (iterable) -> A numpy array storing parameters.
        
        '''
        length = len(self.mapped_data.columns) - 1
        self.params = np.zeros(length, dtype = float)
                
        
    def fit(self):
        '''
        This function fits the given data into the linear regression function. This is the main function of the LR
        model and is not implemented implicitly within the code, instead, it is only implemented when it is called 
        by the LR object
        
        
        '''
        self.set_params()
        if self.optimization == 0:
            sgd_optim = sgd.SGD(params = self.params, data = self.mapped_data, epoch = self.epoch, lr = self.lr, activation = 0, cost = self.cost, huber_point = self.h_p, quantile = self.q )
            self.params = sgd_optim.updated_params
        if self.optimization == 1:
            Adagrad_optim = Adagrad.ADAGRAD(params = self.params, data = self.mapped_data, epoch = self.epoch, lr = self.lr, activation = 0, cost = self.cost, ep = self.ep, huber_point = self.h_p, quantile = self.q)
            self.params = Adagrad_optim.updated_params
        if self.optimization == 2:
            Adadelta_optim = Adadelta.ADADELTA(params = self.params, data = self.mapped_data, epoch = self.epoch, lr = self.lr, activation = 0, cost = self.cost, alpha = self.alpha, ep = self.ep, huber_point = self.h_p, quantile = self.q)
            self.params = Adadelta_optim.updated_params
        if self.optimization == 3:
            Adam_optim = Adam.ADAM(params = self.params, data = self.mapped_data, epoch = self.epoch, lr = self.lr, activation = 0, cost = self.cost, ep = self.ep, b1 = self.b1, b2 = self.b2, huber_point = self.h_p, quantile = self.q)
            self.params = Adam_optim.updated_params
            
    def predict(self, test_data):
        '''
        This function produces the list of predictions from the trained model, given the test_data
        Args:
            test_data (DataFrame) -> The test data for the trained model (without the output values in the last column)
        Returns:
            predictions (Iterable) -> Predictions on test set
        
        '''
        mapped_test_data = self.set_test_basis(test_data, basis = self.basis, order = self.polynomial_basis_order)
        no_of_instance = len(mapped_test_data)
        predicted_list = []
        for i in range(no_of_instance):
            instance_array = np.array(mapped_test_data.iloc[i, :])
            predicted_list.append(np.sum(np.multiply(instance_array, self.params)))
        self.predictions = np.array(predicted_list)
        return self.predictions
        

    def get_params(self):
        print("The Parameters are : ", self.params)
        return self.params
        
             
    def score(self, test_y):
        '''
        This function will be used at two points - first it will be used to predict the score of the training data
        And then to predict the score of validation data (or test_data)
        Args:
            test_y (iterable) -> Containing the actual output values of the test set.
        
        '''
        if self.cost == 0:
            self.loss = lf.mean_squared_loss(y = test_y, pred_y = self.predictions)
        if self.cost == 1:
            self.loss = lf.mean_absolute_loss(y = test_y, pred_y = self.predictions)
        if self.cost == 2:
            self.loss = lf.huber_loss(y = test_y, pred_y = self.predictions, t = self.h_p)
        if self.cost == 3:
            self.loss = lf.log_cosh_loss(y = test_y, pred_y = self.predictions)
        if self.cost == 4:
            self.loss = lf.quantile_loss(y = test_y, pred_y = self.predictions, q = self.q)
        
        
    