import pandas as pd
import numpy as np


class CART():
    def __init__(self, data, ):
        '''
        This class will implement the CART DT algo, and predict the class labels for the given test data
        
        Args:
            data (DataFrame) -> Containing the training data, with the output column at the last
            
        
        '''
        
        
    
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
        
        '''
        
        
        
    def attribute_selector(self, ):
        '''
        This is the main function which is iteratively called to decide on the selection of attribute from the remaining set of attributes during the training
        process
        This function is the one that decides how to treat each attribute value as well, ie (continuous, ordinal, or nominal)
        
        
        '''
        
        
        
    
    
    def fit(sef, ):
        '''
        This is the main training function which trains the DT.
        It creates and implements the primary data structure for housing the DT mdoel.
        It checks (perhaps with another fucntion), to decide if the splitting criteria has been met, and when to stop going deeper
        It calls the attribute_selector(), at each iteration of data.
        It recurssively calls itself as it goes deeper and deeper into the tree
        
        
        '''
        
        
        
        
    def predict(self, test_data, justify = False):
        '''
        This function uses the DT model created by fit() to predict the class label for each instance of the test_data
        If justify is passed to be True, then it also gives for a justification for the class label, by presenting the IG scores for each attribute
        
        
        '''
        
        
        
        
    
    def justify_instance(self, ):
        '''
        This function is called by the predict function and it calls upon self.model_params, to give justification for the assigned class label 
        on the data instance under consideration
        
        
        '''
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    