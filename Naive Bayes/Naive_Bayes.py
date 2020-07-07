# I am gonna code a simple naive bayes algorithm first and then try to work my way through the one with the assumption that P(X|y) is Gaussian

# One thing to keep in mind here. Just like we were finding the overall likelihood of the dataset and then producing the log likelihood out of it, 
# and then using gradient descent to find the optimal values, but we didn't implement any of the mathematical part. We calculated it manually and
# then coded the model with the close form solution. Just like that we will not be 'finding' the parameter values, instead we will derive the formulae
# for evaluation and then just code that formulae here.

import pandas as pd
import numpy as np

class BinomialNB():
    def __init__(self, data, make_discrete = True):
        '''
        This class wil produce the classification predictions while making the naive assumption that the current probability of x, given an output y,
        does not depend on the earlier values of x or y and only depends on the current value of y. This way, each "cluster" will have its own distribution 
        of x, independent of the distribution values of other clusters. 
        
        In Binomial NB, we assume that all the data is categorical with only two categories. The output column, however, can have multiple classes.
        
        
        '''
        self.params = None
        self.data = data
        self.make_discrete = make_discrete
        self.check_discrete()
        self.initiate_params()
        self.predictions = None # This will house the prediction values for all the 'possible' y values.
        
        
    
    def check_discrete(self):
        '''
        This function checks whether the given data is discreet or not. It then converts the data into discreet values if it is not already. 
        Here, since its a Binomial NB, I'll find the mean of the range and then discretize the data based on whether it is smaller or larger than the mean.
        This is necessary for the simple NB model that I am making. Here, the param values are to be derived using simple algebric expressions.
        
        
        '''
        for j in range(len(self.data.columns) - 1):
            if len(self.data.iloc[:, j].unique) != 2:
                col_mean = np.mean(self.data.iloc[:,j])
                for i in range(len(self.data.iloc[:, j])):
                    if self.data.iloc[i,j] > col_mean:
                        self.data.col.iloc[i,j] = 1
                    else:
                        self.data.col.iloc[i,j] = 0
    
    
    
    def initiate_params(self):
        '''
        This function initiates the parameters (based on the number of output values in the given data). 
        Depending on the value of the distribution variable, the parameters defined might be describing a simple, gaussian, poisson, laplace, etc. distribution
        So, if make_discrete is false, and distribution != 0, then we will initiate those particular parameters as per the distribution defined.
        
        The best way I see to do this is by creating a 3D numpy array, with x columns on the x-axis and y values on the y axis, and possible x values in z axis
        Each element of the array would house the probabilities created using the simple y-values is this "AND" x-value is this.
        
        Returns:
            self.params_xy (numpy array) -> A 3D array containing the parameter values
            self.params_y (numpy array) -> A 1D array to contain the paramater values for each individual output class
        '''
        self.x_count = len(self.data.columns) - 1
        self.y_count = len(self.data.iloc[:, -1].unique)
        self.x_3d = 2
        y_keys = list(self.data.iloc[:, -1].unique)
        y_values = list(range(len(y_keys)))
        self.y_dict = {y_keys[i]: y_values[i] for i in range(len(y_keys))}
        x_keys = []
        x_values = [0, 1]
        self.x_dict = []
        for i in range(self.x_count):
            x_keys.append(list(self.data.iloc[:, i].unique))
            self.x_dict.append({x_keys[j]: x_values[j] for j in range(len(x_keys))})
            
        self.params_xy = np.zeros((self.y_count, self.x_count, self.x_3d))
        self.params_y = np.zeros(self.y_count)
        self.calculate_binomial_params() 
        
      
        
    def calculate_binomial_params(self):
        '''
        This function calculates the parameter values using a simple maximum likelihood definition on the joint likelihood definition of the dataset.
        It returns phi(j|y=0), phi(j|y = 1), phi(j|y = 2), etc.
        
        This function will be called from the function initiate_params()
        '''
        self.output = self.data.iloc[:, -1]
        total_len = len(self.data)
        for i in range(self.y_count):
            for item  in self.y_dict.items():
                if item[1] == i:
                    y_key = item[0]
        
            temp_data = self.data.iloc[self.data.iloc[:, -1] == y_key, :]
            for j in range(self.x_count):
                k = 0 # Here, I'll fix this value to 1, because its a binomial distribution. In other cases, this would be a loop
                for item in self.x_dict[j].items():
                    if item[1] == k:
                        x_key = item[0]
                self.params_xy[i, j, k]  = (len(temp_data.iloc[:, j] == x_key))/(len(temp_data))
                self.params_xy[i, j, 1] = 1 - self.params_xy[i, j, k]   # We wouldn't have needed this line if we were looping over all the values of k  
               
            self.param_y[i] = len(temp_data)/total_len
    
    
    
    def get_params(self):
        print('The parameter values are', self.params)
        return self.params
        
        
        
    def make_predictions(self, test_data):
        '''
        This function will take in the test_data, and produce the highest probable class for each instance of data fed.
        It will do so, by producing probabilities for all the possible output classes of y, and storing the data in self.predictions.
        From this iterable array, it will then produce the class with the highest probability along with its probability
        
        '''
        
        
        
        
        
        
        
class MultinomialNB():
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
        This function checks whether the given data is discreet or not. It then converts the data into discreet values if it is not, it will turn the contnuous
        data to discrete data by interepreting it in multinomial form (with predefined number of bins/ the no. suitable for the given range). 
        This is necessary for the simple NB model that I am making. Here, the param values are to be derived using simple algebric expressions.
        
        
        '''
        # I still have to add the bins system to it. That is, the data will be discretized based on the number of bins.
        for j in range(len(self.data.columns) - 1):
            if len(self.data.iloc[:, j].unique) != 2:
                col_mean = np.mean(self.data.iloc[:,j])
                for i in range(len(self.data.iloc[:, j])):
                    if self.data.iloc[i,j] > col_mean:
                        self.data.col.iloc[i,j] = 1
                    else:
                        self.data.col.iloc[i,j] = 0
    
    
    
    
    
    def initiate_params(self):
        '''
        This function initiates the parameters (based on the number of output values in the given data). 
        Depending on the value of the distribution variable, the parameters defined might be describing a simple, gaussian, poisson, laplace, etc. distribution
        So, if make_discrete is false, and distribution != 0, then we will initiate those particular parameters as per the distribution defined.
        
        
        '''
        
        
                        
        self.x_count = len(self.data.columns) - 1
        self.y_count = len(self.data.iloc[:, -1].unique)
        self.x_3d = 2
        y_keys = list(self.data.iloc[:, -1].unique)
        y_values = list(range(len(y_keys)))
        self.y_dict = {y_keys[i]: y_values[i] for i in range(len(y_keys))}
        x_keys = []
        self.x_dict = []
        for i in range(self.x_count):
            x_keys.append(list(self.data.iloc[:, i].unique))
            x_values = list(range(len(x_keys[-1]))) # Have to make sure that this part of the code is correct
            self.x_dict.append({x_keys[j]: x_values[j] for j in range(len(x_keys))})
            
        self.params_xy = np.zeros((self.y_count, self.x_count, self.x_3d))
        self.params_y = np.zeros(self.y_count)
        self.calculate_multinomial_params() 

      
        
        
        
        
        
    def calculate_multinomial_params(self):
        '''
        This function calculates the parameter values using a simple maximum likelihood definition on the joint likelihood definition of the dataset.
        It returns phi(j|y=0), phi(j|y = 1), phi(j|y = 2), etc.
        
        This function will be called from the function all_params()
        ''' 
        self.output = self.data.iloc[:, -1]
        total_len = len(self.data)
        for i in range(self.y_count):
            for item  in self.y_dict.items():
                if item[1] == i:
                    y_key = item[0]
        
            temp_data = self.data.iloc[self.data.iloc[:, -1] == y_key, :]
            for j in range(self.x_count):
                k = 0 # Here, I'll fix this value to 1, because its a binomial distribution. In other cases, this would be a loop
                for item in self.x_dict[j].items():
                    if item[1] == k:
                        x_key = item[0]
                self.params_xy[i, j, k]  = (len(temp_data.iloc[:, j] == x_key))/(len(temp_data))
                self.params_xy[i, j, 1] = 1 - self.params_xy[i, j, k]   # We wouldn't have needed this line if we were looping over all the values of k  
               
            self.param_y[i] = len(temp_data)/total_len
        
        
        
        
    def make_predictions(self, test_data):
        '''
        This function will take in the test_data, and produce the highest probable class for each instance of data fed.
        
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
        
        
        
        
        
    def make_predictions(self, test_data):
        '''
        This function will take in the test_data, and produce the highest probable class for each instance of data fed.
        
        '''
        
        
        
        
        
        
        
        
        
        
        
        
        