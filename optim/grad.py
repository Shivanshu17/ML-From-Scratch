import numpy as np
import pandas as pd
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
    def __init__(self, cost = 0, data, params):
        self.cost = cost
        self.data = data
        self.params = params
        
    def grad_mse(self):
        '''
        This function derives the gradient of cost functions with mean squared error.
        
        
        '''
        
    
    if __name__ == "__main__":
        if cost == 0:
            gradient = grad_mse()
        if cost == 1:
            gradient = grad_mae()
        if cost == 2:
            gradient = grad_huber()
        if cost == 3:
            gradient = grad_log_cosh()
        if cost == 4:
            gradient = grad_quantile()
        return gradient