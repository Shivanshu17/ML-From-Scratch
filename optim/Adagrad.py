from grad_loss import *
import numpy as np
import pandas as pd
class Adagrad():
    def __init__(self, params, lr = 0.1, epoch = 50, data, cost = 0, ep = 1e-8, activation = 0, huber_point = 0.01):
        '''
        The following costs imply the following loss functions:
            0 - Mean Squared Error
            1 - Mean Absolute Error
            2 - Huber Loss
            3 - Log Cosh loss function
            4 - Quantile Error
            
        Ideally the epochs should be contained within a container class. This will make modifications lot easier, but this approach 
        will do for now.
        '''
        if lr < 0.0:
            raise ValueError("Invalid Learning Rate")
        if epoch < 1:
            raise ValueError("Invalid epoch value")
        if cost<0 or cost >4:
            raise ValueError("Invalid cost value, it should be between 0 and 4")
        if activation<0 or activation >4:
            raise ValueError("Activation value should be between 0 and 4")
        defaults = dict(lr = lr, cost = cost, epoch = epoch)
    
        
    def adagrad(self):
        
        '''
        This function optimizes the paramters using gradient function and performs the action 'epoch' number of times.
        
        Variable:
            squared_gradient -> Numpy Array
        Returns:
            params -> Numpy Array representing the updated parameters.
            
        '''
        gradient = 0
        sum_squared_gradient = 0
        for i in range(defaults[epoch]):
            gradient = grad_loss(cost, data, params, activation)
            sum_squared_gradient = sum_squared_gradient + squared_grad(gradient)
            val = np.sqrt(sum_squared_gradient + ep)
            for i in range(len(params)):
                params[i] = params[i] - ((lr/val[i])*gradient[i])
        return params
    
    def squared_grad(self, gradient):
        
        '''     
        This function returns the squared values of each gradient
        
        Args:
            gradient -> A numpy array containing gradients of all parameters.
            
        Returns:
            squared_gradient -> A numpy array such that each element has been squared.
        
        '''
        gradient2 = np.copy(gradient)
        squared_gradient = np.multiply(gradient, gradient2)
        return squared_gradient
    
    if __name__ == "__main__":
        params = adagrad()
        return params