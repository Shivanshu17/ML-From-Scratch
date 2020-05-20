import pandas as pd
import numpy as np
import math
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
            if np.sum(np.multiply(instance_array, params)) < d_p:
                activated_value.append(0)
            else:
                activated_value.append(1)
        f_x = np.array(activated_value)
        return f_x

            
            
     def sigmoid_activation(self):
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
        for i in range(no_of_instance):
            instance_array = np.array(data[i])
            param_value = np.sum(np.multiply(instance_array, params))
            exp_param_value = math.exp(param_value)
            activated_value.append((1/(1+exp_param_value)))
        f_x = np.array(activated_value)
        return f_x
        
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
        for i in range(no_of_instance):
            instance_array = np.array(data[i])
            param_value = np.sum(np.multiply(instance_array, params))
            exp_param_value = math.exp(param_value)
            neg_exp = math.exp(-1*param_value)
            activated_value.append((exp_param_value - neg_exp)/(exp_param_value + neg_exp))
        f_x = np.array(activated_value)
        return f_x
    
    
    def relu_activation(self):
        '''
         This function will return the relu activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
         no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if len(params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            isntance_array = np.array(data[i])
            param_value = np.sum(np.multiply(instance_array, params))
            if param_value>0:
                activated_value.append(param_value)
            else:
                activated_value.append(0)
        f_x = np.array(activated_value)
        return f_x
    
    
    def leaky_relu_activation(self, alpha):
        '''
         This function will return the leaky relu activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(data)
        no_of_cols = len(data.columns)
        activated_value = []
        if alpha>=1:
            raise ValueError("Alpha for leaky relu should be less than 1")
        if len(params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            isntance_array = np.array(data[i])
            param_value = np.sum(np.multiply(instance_array, params))
            if param_value>0:
                activated_value.append(param_value)
            else:
                activated_value.append(alpha * param_value)
        f_x = np.array(activated_value)
        return f_x
    
    
    
    
        
        
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
        