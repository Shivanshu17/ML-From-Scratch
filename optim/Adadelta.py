import grad_loss
import numpy as np
import pandas as pd
class ADADELTA():
    def __init__(self, params,  data, lr = 0.1, epoch = 50, cost = 0, ep = 1e-8, activation = 0, huber_point = 0.0, alpha = 0.9, quantile = 0):
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
        self.activation = activation
        self.params = params
        self.lr = lr
        self.epoch = epoch
        self.data = data
        self.cost = cost
        self.ep = ep
        self.huber_point = huber_point
        self.alpha = alpha
        self.quantile = quantile
        self.updated_params = self.adadelta()
        # defaults = dict(lr = lr, cost = cost, epoch = epoch)
    
        
    def adadelta(self):
        
        '''
        This function optimizes the paramters using gradient function and performs the action 'epoch' number of times.
        
        Variable:
            squared_gradient -> Numpy Array
        Returns:
            params -> Numpy Array representing the updated parameters.
            
        '''
        gradient = 0
        sum_squared_gradient = 0
        for i in range(self.epoch):
            g_obj = grad_loss.grad(cost = self.cost, data = self.data, params = self.params, activation = self.activation, h_p = self.huber_point,  q = self.quantile)
            gradient = g_obj.gradient
            sum_squared_gradient = self.alpha * sum_squared_gradient + (1-self.alpha) * self.squared_grad(gradient)
            val = np.sqrt(sum_squared_gradient + self.ep)
            for i in range(len(self.params)):
                self.params[i] = self.params[i] - ((self.lr/val[i])*gradient[i])
        return self.params
    
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
   
    '''
    if __name__ == "__main__":
        params = adadelta()
        return params
    '''