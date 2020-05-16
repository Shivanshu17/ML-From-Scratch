import numpy as np
class grad():
    '''
    This class will serve to return the gradients of various cost functions as used within various optimisation algorithms.
    I will have to manually derive the gradient formulas and code the functions such that the process can either be vectorised
    (ideal scenario) or each individual gradient can effectively be derived. I will have to handle the X, Y, H(x) and therefore 
    apply some basic data manipulation operations on the 'data' parameter as well.
    
    gradient -> Numpy array containing the gradients of all the parameters.
    '''
    def __init__(self, cost = 0, data, params):
        self.cost = cost
        self.data = data
        self.params = params
        
    def grad_mse(self):
        
    
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