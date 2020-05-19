import pandas as pd
import numpy as np
class activation_function():
    ''' 
    This class will produce activations of the input data
    
    Input:
        data (dataframe) -> A normalised dataframe containing all the input data
        params (iterable) -> Iterable containing the parameter values
        activation (int) -> Number determining the activation function to be employed:
            0  -  Linear Activation
            1  -  Binary Activation
            2  -  Sigmoid Activation
            3  -  Tanh Activation
            4  -  ReLU Activation
            5  -  Leaky ReLU Activation
            6  -  SiLU Activation
            7  -  Softmax Function
        
    Returns:
        f_x (iterable) -> Representing an numpy array of m activation values - one for each instance.
    
    
    '''
    
    def __init__(self, data, params, activation = 0):
        self.data = data
        self.params = params
        self.activation = activation
        
    def linear_activation(self):
        '''
        This function will find linear activations of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
            
        '''
        no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if len(params)!= no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(data[i])
            activated_value.append(np.sum(np.multiply(instance_array, params)))
        f_x = np.array(activated_value)
        return f_x
    
    def binary_activation(self, d_p):
        '''
        This function will return the binary activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if len(params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(data[i])            

            
            
     def sigmoid_activation(self, d_p):
        '''
        This function will return the sigmoid activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if len(params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        
     def tanh_activation(self, d_p):
        '''
        This function will return the tanh activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if len(params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
            
        
        
    
        
        
    if __name__ == "__main__":
        if activation == 0:
            f_x = linear_activation()
        if activation == 1:
            f_x = binary_activation(divide_point)
        if activation == 2:
            f_x = sigmoid_activation()
        if activation == 3:
            f_x = tanh_activation()
        if activation == 4:
            f_x = relu_activation()
        if activation == 5:
            f_x = leaky_relu_activation()
        if activation == 6:
            f_x = silu_activation()
        if activation == 7:
            f_x = softmax_activation()
        