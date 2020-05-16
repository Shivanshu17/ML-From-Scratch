import torch
from grad import *
import numpy as np
class Adagrad():
    def __init__(self, params, lr = 0.1, epoch = 50, data, cost = 0, ep = 1e-8):
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
        defaults = dict(lr = lr, cost = cost, epoch = epoch)
    
        
    def adagrad(self):
        
        '''
        Again, special care will have to be taken at this stage as well, to ensure that the vectorised updates, that is,
        each parameter (among the 'params') is updated according to its own modified learning rate, which will be different
        for each parameter.
        
        
        squared_gradient -> Numpy Array
        
        '''
        gradient = 0
        sum_squared_gradient = 0
        for i in range(defaults[epoch]):
            gradient = grad(cost, data, params)
            sum_squared_gradient = sum_squared_gradient + squared_grad(gradient)
            val = np.sqrt(sum_squared_gradient + ep)
            '''
            if cost == 0:
                gradient = grad.grad_msd(data, params)
            if cost == 1:
                gradient = grad.grad_msa(data, params)
            if cost == 2:
                gradient = grad.grad_huber(data, params)
            if cost == 3:
                gradient = grad.grad_log_cosh(data, params)
            if cost == 4:
                gradient = grad.grad_quantile(data, params)
            '''
            for i in range(len(params)):
                params[i] = params[i] - ((lr/val[i])*gradient[i])
        return params
    
    def squared_grad(self, gradient):
        
        '''
        This function will derive the squared values of all the individual gradients, it has to do so, such that the 
        array form of the data is maintained. Something about the 'units' and 'vector product' was mentioned in the approach
        nuances and will have to be ensured in this function.
        
        '''