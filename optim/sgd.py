import torch
from grad_loss import *
import numpy as np
import pandas as pd
class SGD():
    def __init__(self, params, lr = 0.1, nesterov = False, momentum = 0, epoch = 50, data, cost = 0):
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
        if momentum < 0:
            raise ValueError("Invalid momentum value")
        if weight_decay < 0:
            raise ValueError("Invalid weight decay value")
        if epoch < 1:
            raise ValueError("Invalid epoch value")
        if cost<0 or cost >4:
            raise ValueError("Invalid cost value, it should be between 0 and 4")
        defaults = dict(lr = lr, nesterov = nesterov, momentum = momentum, weight_decay = weight_decay, cost = cost, epoch = epoch)
    
        
    def sgd(self):
        '''
        
        
        
        '''
        gradient = 0
        momentum_term = 0
        for i in range(defaults[epoch]):
            if defaults[nesterov] == True:
                future_params = params - (defaults[momentum]*momentum_term)
                gradient = grad_loss(cost, data, future_params)
            else:
                gradient = grad(cost, data, params)
            momentum_term = (defaults[momentum] * momentum_term)  + (defaults[lr]*gradient)
            params = params - momentum_term
        return params
            
    if __name__ == "__main__":
        params = sgd()
        return params