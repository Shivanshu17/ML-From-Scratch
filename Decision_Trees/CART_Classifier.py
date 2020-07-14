import pandas as pd
import numpy as np


class CART():
    def __init__(self, data, cost = 1, max_depth = 3):
        '''
        This class will implement the CART DT algo, and predict the class labels for the given test data
        
        Args:
            data (DataFrame) -> Containing the training data, with the output column at the last.
            cost (int) -> Determines the loss function to be employed:
                0: Entropy Loss
                1: Gini Index
                2: Classification Loss
        
        '''
        self.data = data #It assumes that the data is numeric (even for categorical data, that it has been preprocessed to convert it into form of categories)
        self.all_attributes = None
        self.model_params = None
        self.class_labels = None
        self.cost = cost
        self.max_Depth = max_depth
    
    
    
    
    def ordinal_permutations(self, ):
        '''
        This function will produce all the possible permutations for binary split of ordinal data
        
        '''
        
        
        
        
    
    def nominal_permutations(self, ):
        '''
        This function produces all the nominal permuations for binary split pairs of the nominal data column
        
        '''
        
        
        
        
        
    
    
    def continous_permutatons(self, ):
        '''
        This function discretizes the continuous data and produces the permutations by treating the produced classifications as ordinal data
        
        '''
        
        
        
        
        
    def gini_loss(self, ):
        '''
        This function calculates the impurity percentage of the attribute split with gini_index
        
        
        '''
        
        
        
        
        
    
    def entropy_loss(self, ):
        '''
        This function calculates the impurity percentage of the attribute split with entropy_loss
        
        
        '''
        
        
        
        
        
    def classification_error(self, ):
        '''
        This function calculates the impurity metric of the attribute using simple classification error
        
        
        '''
        
        
        
        
        
        
    
    def information_gain(self, ):
        '''
        This function is called from the attribute_selector function (for each attribute). It calculates the information gain of the split according to 
        a particular attribute. This function calls the loss_functions, based on the loss metric prescribed by the user.
        
        We don't have to calculate the gain ratio in CART (because splits are limited to binary splits), but we might need to implement it for C4.5 and others
        '''
        
        
        
    
    
    
    
    def attribute_class_label(self, ):
        '''
        This function would assign the class label to the nodes that have met the "stopping_condition".
        It will also have to take into consideration various scenarios like when a node might not even have a single instance, etc.
        
        After having sorted through the scenarios and settling for a class label, it is then stored in the 'class_label' column of the self.model_params
        
        
        '''
        
        
        
        


    # Maybe I should also add a datset transformation function which will imply the type of each column (nominal, ordinal or continuous), and store the data from
    # continuous columns in the discretized format (so that we don't have to discreticize the colummns at each call to attribute selector).
    
        
    def attribute_selector(self, df, attribute_list ):
        '''
        This is the main function which is iteratively called to decide on the selection of attribute from the remaining set of attributes during the training
        process
        This function is the one that decides how to treat each attribute value as well, ie (continuous, ordinal, or nominal)
        
        This is like the "find_best_split" function definition of the algorithm defined in the book
        
        Args:
            df (dataframe) -> The Dataframe to perform attribute selection based upon.
            attribute_list (iterable) -> Contains the names of the columns from which the ideal attribute for the node is to be selected.
        
        '''
        
        
        
        
        
    
    
    def check_split_criteria(self, ):
        '''
        This function is called upon by fit module at each node and it decides whether or not the splitting conditions have been met.
        
        Returns:
            stop_splits (Boolean) -> True, if the stopping criteria is met and we do not want to fragmentize further
        
        
        '''
        
        
        
        
        
    def create_node(self, ):
        '''
        This function creates the node in the self.model_params dataframe. This function is optional, and I will try to include it within fit function
        
        '''
        
        
        
        
        
    
    def fit(self, ): # Will have to change the implementation to recursive definition
        '''
        This is the main training function which trains the DT.
        It creates and implements the primary data structure for housing the DT mdoel.
        It checks (perhaps with another fucntion), to decide if the splitting criteria has been met, and when to stop going deeper
        It calls the attribute_selector(), at each iteration of data. It will also pass the list of attributes to select from. 
        It recurssively calls itself as it goes deeper and deeper into the tree (this step is optional, and lets see if we can perform this with iteration)
        
        
        
        
        '''
        self.model_params = pd.DataFrame(columns = ['attribute_name', 'level', 'attribute_type', 'condition','child_index', 'parent_index', 'class_label']) # Attribute_type will store (0, 1, 2, 3) for binomial, nominal, ordinal, and continuous data
        self.all_attributes = self.data.columns[:-1]
        self.y_count = len(self.data.iloc[:, -1].unique)
        y_keys = list(self.data.iloc[:, -1].unique)
        self.y_dict = {y_keys[i]: y_values[i] for i in range(len(y_keys))}
        self.class_labels = y_keys
        stop_splits = self.check_split_criteria() # Will probably have to pass quite a few details here as arguments, will see later...
        for i in range(len(self.all_attributes)):
            
            if stop_splits:
                # Code for assigning class labels
                # I will conclude the class lable and add it to the self.model_params
                # It will also remove the attribute under consideration from the 'current' list of attributes
                
            else:
                
            
        
        
        
        
        
        
        
        
    def predict(self, test_data, justify = False):
        '''
        This function uses the DT model created by fit() to predict the class label for each instance of the test_data
        If justify is passed to be True, then it also gives for a justification for the class label, by presenting the IG scores for each attribute
        
        
        '''
        
        
        
        
    def predict_probability(self, test_data):
        '''
        This function first calculates the predicted class for the given item (by calling predict()), and then based on that class, it returns the probability
        of the data being in that class as equal to the ratio of instances of that class in the training data over the total number of instances.
        
        '''
        
        
        
        
    
    def justify_instance(self, ):
        '''
        This function is called by the predict function and it calls upon self.model_params, to give justification for the assigned class label 
        on the data instance under consideration
        
        
        '''
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    