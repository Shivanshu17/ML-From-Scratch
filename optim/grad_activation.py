import pandas as pd
import numpy as np
import math
class grad_activation():
    ''' 
    This class will produce gradients of the activations of the input data
    
    Input:
        data (dataframe) -> A normalised dataframe containing all the input data
        params (iterable) -> Iterable containing the parameter values
        activation (int) -> Number determining the activation function for which the gradient is to be determined:
            0  -  Linear Activation
            1  -  Binary Activation
            2  -  Sigmoid Activation
            3  -  Tanh Activation
            4  -  ReLU Activation
            5  -  Leaky ReLU Activation
            6  -  SiLU Activation
            7  -  Softmax Function
        
    Returns:
        g_activation (iterable) -> Representing an numpy array of m activation values - one for each instance.
    
    
    '''    
    def __init__(self, data, params, activation = 0):
        self.data = data
        self.params = params
        self.activation = activation
        
    
    def linear_grad(self):
        
    
    
    if __name__ == "__main__":
        if activation == 0:
            g_activation = linear_grad()
        if activation == 1:
            g_activation = binary_grad()
        if activation == 2:
            g_activation = sigmoid_grad()
        if activation == 3:
            g_activation = tanh_grad()
        if activation == 4:
            g_activation = relu_grad()
        if activation == 5:
            g_activation = leaky_relu_grad()
        if activation == 6:
            g_activation = silu_grad()
        if activation ==7:
            g_activation = softmax_grad()
        return g_activation
    