import pandas as pd
import numpy as np
from scipy.stats import norm
import math  

class GaussianNB():
    # Here is the benefit of coding the algorithms myself. So consider the scenario when we are using a Generatvie model, but Gaussian distribution is unable
    # to map all the classes. It might be good for one or two of the classes, and (based on the data), some other function would probably fit those classes better
    # Now, this way, we could code different distributions for each class and therefore, provide our model with more capabilities.
    def __init__(self, data):
        '''
        This class can take in continuous data and produce predictions of the test set. Here we will be assuming that the data is Gaussian in nature
        and will train our parameters accordingly
        
        '''
        self.params_xy = None
        self.params_y = None
        self.data = data
        self.initiate_params()
        
        
        
    def initiate_params(self):
        '''
        This function will initiate the parameter values for the given data. The number of gaussian distributions to be trained is equal to the 
        total number of output classes. 
        
        '''
        self.x_count = len(self.data.columns) - 1
        self.y_count = len(self.data.iloc[:, -1].unique)
        self.x_3d = 2
        y_keys = list(self.data.iloc[:, -1].unique)
        y_values = list(range(len(y_keys)))
        self.y_dict = {y_keys[i]: y_values[i] for i in range(len(y_keys))}
        self.params_xy = np.zeros((self.y_count, self.x_count, self.x_3d))
        self.params_y = np.zeros(self.y_count)
        self.calculate_gaussian_params()
        
        
        
    def calculate_gaussian_params(self):
        '''
        This function calculates the Gaussian parameter values. I have realised that they can be produced just by calculating the mean and variance
        for each class of output. 
        
        
        '''
        total_len = len(self.data)
        for i in range(self.y_count):
            for item in self.y_dict.items():
                if item[1] == i:
                    y_key = item[0]
            temp_data = self.data.iloc[self.data.iloc[:, -1] == y_key, :]
            for j in range(self.x_count):
                data_to_fit = temp_data.iloc[:, j]
                self.params_xy[i, j, 0] = np.mean(np.array(data_to_fit))
                self.params_xy[i, j, 1] = np.std(np.array(data_to_fit))
                # self.prarams_xy[i, j, 0], self.params_xy[i, j, 1] = norm.fit(data_to_fit)
            self.params_y[i] = len(temp_data)/total_len
                           
        
        
        
    def calculate_gaussian_values(self, mean, std, data):
        '''
        This function would produce the probability values of the gaussian distribution according to its parameters.
        This function will be used by test_data to transform the data, given the gaussian parameters, and will return the probability of the given data
        element to be following the gaussian parameters passed to this function.
        
        '''
        var = float(std)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(data)-float(mean))**2/(2*var))
        # This could be used from the scipy.stats library as well -> norm(mean_value,std_value).pdf(data_value)
        return num/denom
                
    
    def get_params(self):
        print('The parameter values for x & y are', self.params_xy)
        print('The parameter values for y alone (priors) are', self.params_y)
        return self.params_xy, self.params_y

    
    
    def make_predictions(self, test_data):
        '''
        This function will take in the test_data, and produce the highest probable class for each instance of data fed.
        
        '''
        temp_array = np.zeros(self.y_count, 2) # temp_array = np.zeros(self.y_count, self.y_count)
        self.predictions = pd.DataFrame(temp_array, columns = ['y_key', 'y_prob'])
        for i in range(len(test_data)):
            test_row_y_probs =  np.ones(self.y_count)
            for j in range(self.y_count):
                prob_value = 1.0
                for k in range(self.x_count):
                    data_value = test_data.iloc[i, k]
                    mean_value = self.params_xy[j, k, 0]
                    std_value = self.params_xy[j, k, 1]
                    prob_value = prob_value * self.calculate_gaussian_values(mean = mean_value, std = std_value, data = data_value)
                test_row_y_probs[j] = test_row_y_probs[j] * prob_value * self.params_y[j]
            test_row_total_prob = np.sum(test_row_y_probs)
            test_row_y_probs = test_row_y_probs/test_row_total_prob
            test_row_class_prob = np.max(test_row_y_probs)
            test_row_class_value = np.where(test_row_y_probs == test_row_class_prob)
            for item in self.y_dict.items():
                if item[1] == test_row_class_value:
                    test_row_class_key = item[0]
            test_row_df = pd.DataFrame(data = {'y_key': test_row_class_key, 'y_prob': test_row_class_prob})
            pd.concat([self.predictions, test_row_df], axis = 1)
        return self.predictions
        