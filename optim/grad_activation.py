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
        i (int) -> Represents the parameter point under consideration (from the list of all the parameters)
    Returns:
        g_activation (iterable) -> Representing an numpy array of m activation values - one for each instance.
    
    
    '''    
    def __init__(self, data, params, activation = 0, i):
        self.data = data
        self.params = params
        self.activation = activation
        self.i = i
        if self.activation == 0:
            self.g_activation = self.linear_grad()
        if self.activation == 1:
            self.g_activation = self.binary_grad()
        if self.activation == 2:
            self.g_activation = self.sigmoid_grad()
        if self.activation == 3:
            self.g_activation = self.tanh_grad()
        if self.activation == 4:
            self.g_activation = self.relu_grad()
        if self.activation == 5:
            self.g_activation = self.leaky_relu_grad()
        if self.activation == 6:
            self.g_activation = self.silu_grad()
        if self.activation ==7:
            self.g_activation = self.softmax_grad()
    
    def linear_grad(self):
        i_col = self.data.iloc[:, self.i]
        g_activation = np.array(i_col)
        return g_activation
    
    def binary_grad(self):
        i_col = self.data.iloc[:, self.i]
        g_activation = np.array(i_col)
        g_activation = g_activation * 0
        return g_activation
    
    def sigmoid_grad(self):
        i_col = np.array(self.data.iloc[:, self.i])
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for j in range(no_of_instance):
            instance_array = np.array(self.data[j])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            activated_value.append((1/(1+exp_param_value)*(1+exp_param_value))*exp_param_value)
        g_activation = np.multiply(np.array(activation_value), i_col)
        return g_activation
    
    def tanh_grad(self):
        i_col = np.array(self.data.iloc[:, self.i])
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for j in range(no_of_instance):
            instance_array = np.array(self.data[j])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            square_exp_value = math.exp(2*param_value)
            activated_value.append((1/(1+exp_param_value)*(1+exp_param_value))*2*square_exp_value)
        g_activation = np.multiply(np.array(activation_value), i_col)
        return g_activation
    
    def relu_grad(self):
        i_col = np.array(self.data.iloc[:, self.i])
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        grad_activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for j in range(no_of_instance):
            instance_array = np.array(self.data[j])
            param_value = np.sum(np.multiply(instance_array, self.params))
            if param_value>0:
                grad_activated_value.append(1)
            else:
                grad_activated_value.append(0)
        g_activation = np.array(grad_activated_value)
        return g_activation
    
    def leaky_relu_grad(self):
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        grad_activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for j in range(no_of_instance):
            instance_array = np.array(self.data[j])
            param_value = np.sum(np.multiply(instance_array, self.params))
            if param_value>0:
                grad_activated_value.append(1)
            else:
                grad_activated_value.append(0.1)
        g_activation = np.array(grad_activated_value)
        return g_activation
    
    def silu_grad(self):
        i_col = np.array(self.data.iloc[:, self.i])
        no_of_instance = len(self.data)
        no_of_cols = len(self.data.columns)
        activated_value = []
        if len(self.params) != no_of_cols:
            raise ValueError("The number of parameters should be equal to the number of columns")
        for i in range(no_of_instance):
            instance_array = np.array(self.data[i])
            param_value = np.sum(np.multiply(instance_array, self.params))
            exp_param_value = math.exp(param_value)
            activated_value.append(param_value*((1+ (2*exp_param_value)/((1+exp_param_value)*(1+exp_param_value))))
        g_activation = np.multiply(np.array(activation_value), i_col)
        return g_activation
    
'''    
    if __name__ == "__main__":
        if self.activation == 0:
            g_activation = linear_grad()
        if self.activation == 1:
            g_activation = binary_grad()
        if self.activation == 2:
            g_activation = sigmoid_grad()
        if self.activation == 3:
            g_activation = tanh_grad()
        if self.activation == 4:
            g_activation = relu_grad()
        if self.activation == 5:
            g_activation = leaky_relu_grad()
        if self.activation == 6:
            g_activation = silu_grad()
        if self.activation ==7:
            g_activation = softmax_grad()
        return g_activation
''' 