import grad_loss
import numpy as np
import pandas as pd
class SGD():
    def __init__(self, params, data, activation = 0, lr = 0.1, nesterov = False, momentum = 0, epoch = 50, cost = 0, huber_point = 0.0, quantile = 0):
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
        if epoch < 1:
            raise ValueError("Invalid epoch value")
        if cost<0 or cost >4:
            raise ValueError("Invalid cost value, it should be between 0 and 4")
        self.activation = activation
        self.params = params
        self.lr = lr
        self.nesterov = nesterov
        self.momentum = momentum
        self.epoch = epoch
        self.data = data
        self.cost = cost
        self.huber_point = self.huber_point
        self.quantile = quantile
        #defaults = dict(lr = lr, nesterov = nesterov, momentum = momentum, weight_decay = weight_decay, cost = cost, epoch = epoch)
        
        
    def sgd(self):
        '''
        This function calculates the optimized values of parameters by employing stochastic gradient descent.
        It checks whether or not the user wants nesterov acceleration component to be used.
        And a zero value for momentum would mean no use of momentum term as well.
        
        '''
        gradient = 0
        momentum_term = 0
        for i in range(self.epoch):
            if self.nesterov == True:
                future_params = self.params - (self.momentum*momentum_term)
                g_obj = grad_loss(cost = self.cost, data = self.data, params = future_params, activation = self.activation, h_p = self.huber_point, q = self.quantile)
                gradient = g_obj.gradient
            else:
                g_obj = grad_loss(cost = self.cost, data = self.data, params = self.params, activation = self.activation, h_p = self.huber_point, q = self.quantile)
                gradient = g_obj.gradient
            momentum_term = (self.momentum * momentum_term)  + (self.lr*gradient)
            self.params = self.params - momentum_term
        return self.params
            
    if __name__ == "__main__":
        params = sgd()
        return params