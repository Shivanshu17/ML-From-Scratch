# I am gonna code a simple naive bayes algorithm first and then try to work my way through the one with the assumption that P(X|y) is Gaussian

# One thing to keep in mind here. Just like we were finding the overall likelihood of the dataset and then producing the log likelihood out of it, 
# and then using gradient descent to find the optimal values, but we didn't implement any of the mathematical part. We calculated it manually and
# then coded the model with the close form solution. Just like that we will not be 'finding' the parameter values, instead we will derive the formulae
# for evaluation and then just code that formulae here.

import pandas as pd
import numpy as np

class CategoricalNB():
    def __init__(self, data, make_discrete = True):
        '''
        This class wil produce the classification predictions while making the naive assumption that the current probability of x, given an output y,
        does not depend on the earlier values of x or y and only depends on the current value of y. This way, each "cluster" will have its own distribution 
        of x, independent of the distribution values of other clusters. 
        
        
        '''
        self.params = None
        self.data = data
        self.make_discrete = make_discrete
        self.is_discrete = None
        
        
        
    
    def check_discrete(self):
        '''
        This function checks whether the given data is discreet or not. It then converts the data into discreet values if it is not (and if make_discrete is True)
        This is necessary for the simple NB model that I am making. Here, the param values are to be derived using simple algebric expressions.
        
        
        '''
    
    
    
    
    
    def initiate_params(self):
        '''
        This function initiates the parameters (based on the number of output values in the given data). 
        Depending on the value of the distribution variable, the parameters defined might be describing a simple, gaussian, poisson, laplace, etc. distribution
        So, if make_discrete is false, and distribution != 0, then we will initiate those particular parameters as per the distribution defined.
        
        
        '''
        
      
        
        
        
        
        
    def calculate_simple_params(self, df):
        '''
        This function calculates the parameter values using a simple maximum likelihood definition on the joint likelihood definition of the dataset.
        It returns phi(j|y=0), phi(j|y = 1), phi(j|y = 2), etc.
        
        This function will be called from the function all_params()
        '''
        
        
        
    
    
    
        
# We could create classes for many such distributiosn including but not limited to Poisson's Distribution, Bernoulli Distribution, etc.        
        

class GaussianNB():
    # Here is the benefit of coding the algorithms myself. So consider the scenario when we are using a Generatvie model, but Gaussian distribution is unable
    # to map all the classes. It might be good for one or two of the classes, and (based on the data), some other function would probably fit those classes better
    # Now, this way, we could code different distributions for each class and therefore, provide our model with more capabilities.
    def __init__(self, data):
        '''
        This class can take in continuous data and produce predictions of the test set. Here we will be assuming that the data is Gaussian in nature
        and will train our parameters accordingly
        
        '''
        
        
        
        
    def initiate_params(self):
        '''
        This function will initiate the parameter values for the given data. The number of gaussian distributions to be trained is equal to the 
        total number of output classes. 
        
        '''
        
        
        
        
        
    def calculate_gaussian_params(self):
        '''
        This function calculates the Gaussian parameter values. I have realised that they can be produced just by calculating the mean and variance
        for each class of output. 
        
        
        '''
        
        
        
        
        
        
        
        
        
        