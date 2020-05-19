import numpy as np
import pandas as pd
from functools import reduce
from grad_activation import *
class grad():
    '''
    This class will produce the gradients of the cost functions in reference to the data provided.
    
    Args:
        cost (int) -> Determines the cost function to be employed according to the following scheme:
            0 - Mean Squared Error
            1 - Mean Absolute Error
            2 - Huber Loss
            3 - Log Cosh loss function
            4 - Quantile Error
        data (DataFrame) -> Represents the data including the instances and the intended output values.
        params (iterable) -> Array to store the parameters of the cost function.
    
    Returns:
        gradient -> Numpy array containing the gradients of all the parameters.
    
    '''
    def __init__(self, cost = 0, data, params, activation = 0):
        self.cost = cost
        self.data = data
        self.params = params
        
    def grad_mse(self):
        '''
        This function derives the gradient of cost functions with mean squared error.
        
        f_x (iterable) -> A 1D numpy array representing the f(x) value of each instance after applying the activation function. 
                          So, each instance will create just one f(x) value.
        output (iterable) -> A 1D numpy array containing the actual output values. These can either be derived by taking out
                             the last row of the data, or if the dataframe allows for it - the .output attribute of the data object.                  
        g_activation (iterable) -> It will store the gradient of activation function (with respect to the data) as returned 
                                    from the 'grad_activation' function.
                                    
        returns:
            gradient (iterable) -> Gradient of the cost function.
        '''
        output = data.output
        f_x = activation_function(data, params, activation)
        g_activation = grad_activation(data, activation)
        param_grad = []
        number_of_instance = len(output)
        for i in range(len(params)):
            param_grad.append((reduce(lambda item1, item2: item1+item2, map(np.multiply((f_x - output),g_activation ))))/(2*number_of_instance))
        gradient = np.array(param_grad)
        return gradient
    
    def grad_mae(self):
        '''
        This function derives the gradient of cost functions with mean absolute error.
        
        f_x (iterable) -> A 1D numpy array representing the f(x) value of each instance after applying the activation function. 
                          So, each instance will create just one f(x) value.
        output (iterable) -> A 1D numpy array containing the actual output values. These can either be derived by taking out
                             the last row of the data, or if the dataframe allows for it - the .output attribute of the data object.                  
        g_activation (iterable) -> It will store the gradient of activation function (with respect to the data) as returned 
                                    from the 'grad_activation' function.
                                    
        returns:
            gradient (iterable) -> Gradient of the cost function.
            
        '''
        output = data.output
        f_x = activation_function(data, params, activation)
        g_activation = grad_activation(data, activation)
        param_grad = []
        number_of_instance = len(output)
        for i in range(len(params)):
            param_grad.append((reduce(lambda item1, item2: item1 + item2, g_activation.tolist()))/number_of_instance)
        gradient = np.array(param_grad)
        return gradient
    
    def grad_huber(self, h_p):
        '''
        This function derives the gradient of the cost function with huber loss.
        
        
        
        returns:
            gradient (iterable) -> Gradient of the cost function.
        
        '''
        output = data.output
        f_x = activation_function(data, params, activation)
        g_activation = grad_activation(data, activation)
        param_grad = []
        number_of_instance = len(output)
        for j in range(len(params)):
            instance_update
            for i in range(number_of_instances):
                if (f_x[i] - y[i] <= h_p):
                    
                
    
    if __name__ == "__main__":
        if cost == 0:
            gradient = grad_mse()
        if cost == 1:
            gradient = grad_mae()
        if cost == 2:
            gradient = grad_huber(huber_point)
        if cost == 3:
            gradient = grad_log_cosh()
        if cost == 4:
            gradient = grad_quantile()
        return gradient