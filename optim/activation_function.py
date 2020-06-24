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
    
    def __init__(self, data, params, activation = 0, divide_point = 1):
        self.data = data
        self.params = params
        self.activation = activation
        self.divide_point = divide_point   # I haven't taken this data as input from grad_loss function (so this can't be changed from calling function)
        if self.activation == 0:
            self.f_x = self.linear_activation()
        if self.activation == 1:
            self.f_x = self.binary_activation(self.divide_point)
        if self.activation == 2:
            self.f_x = self.sigmoid_activation()
        if self.activation == 3:
            self.f_x = self.tanh_activation()
        if self.activation == 4:
            self.f_x = self.relu_activation()
        if self.activation == 5:
            self.f_x = self.leaky_relu_activation()
        if self.activation == 6:
            self.f_x = self.silu_activation()
        if self.activation == 7:
            self.f_x = self.softmax_activation()
        
        
    def linear_activation(self):
        '''
        This function will find linear activations of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
            
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params)!= no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            activated_value.append(np.sum(np.multiply(instance_array, self.params)))
        f_x = np.array(activated_value)
        return f_x
    
    def binary_activation(self):
        '''
        This function will return the binary activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            if np.sum(np.multiply(instance_array, self.params)) < self.divide_point:
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
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            activated_value.append((1/(1+exp_param_value)))
        f_x = np.array(activated_value)
        return f_x
        
    def tanh_activation(self):
        '''
        This function will return the tanh activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
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
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            if param_value>0:
                activated_value.append(param_value)
            else:
                activated_value.append(0)
        f_x = np.array(activated_value)
        return f_x
    
    
    def leaky_relu_activation(self):
        '''
         This function will return the leaky relu activation values of the input data
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            if param_value>0:
                activated_value.append(param_value)
            else:
                activated_value.append(0.1 * param_value)
        f_x = np.array(activated_value)
        return f_x
    
    
    def silu_activation(self):
        '''
        This function will return the silu activaton values of the input data. SiLU was proposed by researchers at Google,
        and has been tauted as better performer than Leaky ReLU in Computer Vision domain
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            activated_value.append(param_value*(1/(1+exp_param_value)))
            
        f_x = np.array(activated_value)
        return f_x
    
    def softmax_activation(self):
        '''
        This function will returnt the softmax activaion values of the input data. This is typically used in the last layer
        of a multiclass classification problem.
        
        Returns:
            f_x (iterable) -> A numpy array of m activation values
        
        '''
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        exp_value = []
        sum_value = 0
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            exp_value.append(exp_param_value)
            sum_value = sum_value + exp_param_value
        exp_value_array = np.array(exp_param_value)
        f_x = exp_value_array/sum_value
        return f_x
        