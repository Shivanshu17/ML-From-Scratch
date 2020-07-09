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
                        self.data.iloc[i,j] = 1
                    else:
                        self.data.iloc[i,j] = 0
    
    
    
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
        print('The parameter values for x & y are', self.params_xy)
        print('The parameter values for y alone (priors) are', self.params_y)
        return self.params_xy, self.params_y
        
      
    
    def check_test_discrete(self, df):
        '''
        This function checks whether the given data is discreet or not. It then converts the data into discreet values if it is not already. 
        Here, since its a Binomial NB, I'll find the mean of the range and then discretize the data based on whether it is smaller or larger than the mean.
        This is necessary for the simple NB model that I am making. Here, the param values are to be derived using simple algebric expressions.
        
        
        '''
        for j in range(len(df.columns) - 1):
            if len(df.iloc[:, j].unique) != 2:
                col_mean = np.mean(df.iloc[:,j])
                for i in range(len(df.iloc[:, j])):
                    if df.iloc[i,j] > col_mean:
                        df.iloc[i,j] = 1
                    else:
                        df.iloc[i,j] = 0
        return df
    
    
        
    def make_predictions(self, test_data):
        '''
        This function will take in the test_data, and produce the highest probable class for each instance of data fed.
        It will do so, by producing probabilities for all the possible output classes of y, and storing the data in self.predictions.
        From this iterable array, it will then produce the class with the highest probability along with its probability
        
        '''
        test_data = self.check_test_discrete(test_data)
        temp_array = np.zeros(self.y_count, 2) # temp_array = np.zeros(self.y_count, self.y_count)
        self.predictions = pd.DataFrame(temp_array, columns = ['y_key', 'y_prob'])
        for i in range(len(test_data)):

            test_row_y_probs =  np.ones(self.y_count)

            for j in range(self.y_count):
                for k in range(self.x_count):
                    data_key = test_data.iloc[i, k]
                    prob_value = 1.0
                    for item in self.x_dict[k].items():
                        if item[0] == data_key:
                            data_value = item[1]
                    prob_value = prob_value * self.params_xy[j, k, data_value]
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
        
        