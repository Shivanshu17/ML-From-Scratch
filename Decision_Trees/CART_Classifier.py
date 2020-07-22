import pandas as pd
import numpy as np


class CART():
    def __init__(self, data, cost = 1, max_depth = 4, minimum_data_points = 50):
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
        self.cost = cost
        self.max_Depth = max_depth
        self.minimum_data_points = minimum_data_points
        self.model_params = pd.DataFrame(columns = ['attribute_name', 'level', 'attribute_type', 'condition','child_index', 'parent_index', 'class_label']) 
        self.model_params['child_index'] = self.model_params['child_index'].astype('object')
        '''
        # Attribute_type will store (0, 1, 2, 3) for binomial, nominal, ordinal, and continuous data (or should it? I think it can be done without it, will have to see)
        # child_index will be a list that contains indexes for all the children of each node. For this CART implementation, each child will house just two values
        # condition should ideally be a list as well. With [len(child_index) - 1] number of values, but, since in case of CART it is 1, we are gonna just take one condition value in
        # level would indicate the level of the node in the tree, and this would primarily be used by check_split_criteria() function
        # If this were C4.5 or other implementation, we would have had to include a 'number_of_split' column as well (For convenience during the inference stage)
        
        '''
        self.all_attributes = self.data.columns[:-1]
        self.attribute_list = self.data.columns[:-1]
        self.y_count = len(self.data.iloc[:, -1].unique)
        y_keys = list(self.data.iloc[:, -1].unique)
        self.y_dict = {y_keys[i]: y_values[i] for i in range(len(y_keys))}
        self.class_labels = y_keys
        self.level = [0]
        self.index = [-1]
        updated_level, updated_index = self.fit(df = self.data, attributes = self.attribute_list) # Have to make sure that this funciton definition is correct.
        if updated_level == 0 and update_index == -1:
            print("Decision Tree training process was successful")
        
        
    
    
    
    
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
        
        
        
        
        
    def gini_loss(self, p):
        '''
        This function calculates the impurity percentage of the attribute split with gini_index
        
        
        '''
        return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
        
        
        
        
    
    def entropy_loss(self, p):
        '''
        This function calculates the impurity percentage of the attribute split with entropy_loss
        
        
        '''
        return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
        
        
        
        
    def classification_error(self, p):
        '''
        This function calculates the impurity metric of the attribute using simple classification error
        
        
        '''
        return 1 - np.max([p, 1 - p])
        
        
        
        
        
    
    def information_gain(self, ):
        '''
        This function is called from the attribute_selector function (for each attribute). It calculates the information gain of the split according to 
        a particular attribute. This function calls the loss_functions, based on the loss metric prescribed by the user.
        
        We don't have to calculate the gain ratio in CART (because splits are limited to binary splits), but we might need to implement it for C4.5 and others
        '''
        
        
        
    
    
    
    
    def node_class_label(self, df):
        '''
        This function would assign the class label to the nodes that have met the "stopping_condition".
        
        After having sorted through the scenarios and settling for a class label, it is then stored in the 'class_label' column of the self.model_params
        
        Args:
            df (DataFrame) -> Dataframe as sent by the fit function with the output class as the last column
        
        
        '''
        output_column = df.iloc[:, -1]
        class_labels = list(output_column.value_counts().index)
        return class_labels[0]
        
        


    # Maybe I should also add a datset transformation function which will imply the type of each column (nominal, ordinal or continuous), and store the data from
    # continuous columns in the discretized format (so that we don't have to discreticize the colummns at each call to attribute selector).
    
        
    def attribute_selector(self, df, attribute_list ):
        '''
        This is the main function which is iteratively called to decide on the selection of attribute from the remaining set of attributes during the training
        process
        This function is the one that decides how to treat each attribute value, ie (continuous, ordinal, or nominal), and then calls the respective function 
        to get binary splits for the values.
        
        With each permutation of each attribute, it calls the function information_gain(), which then in turn calls the gini or entropy based on self.cost
        
        This is like the "find_best_split" function definition of the algorithm defined in the book
        
        Args:
            df (dataframe) -> The Dataframe to perform attribute selection based upon.
            attribute_list (iterable) -> Contains the names of the columns from which the ideal attribute for the node is to be selected.
        
        '''
        
        
        
        
        
    
    
    def check_split_criteria(self, df):
        '''
        This function is called upon by fit module at each node and it decides whether or not the splitting conditions have been met.
        
        Returns:
            stop_splits (Boolean) -> True, if the stopping criteria is met and we do not want to fragmentize further
        
        
        '''
        stop_splits = False
        # To check if the node only contains one class
        no_of_classes = len(pd.unique(df.iloc[:, -1]))
        # If the node contains less than a certain number of data points
        len_of_data = len(df)
        # To check if the maximum depth is reached
        depth = self.level[0]
        # To check if the node purity is sufficient (or node_entropy is too high for the best split). I will most probably call the information_gain() with each individual attribute to check if the best attribute split isn't any good either; The code would probably be largely derived from attribute selector
        # Another approach to node purity might be to evaluate the softmax value of each class, and if the highest value is higher than the node purity we are looking for, then we can stop splitting
        node_purity = df.iloc[:, -1].value_counts(normalize = True).iloc[0] # Still have to make sure the returned value of value_counts is suitable of iloc
        
        if no_of_classes == 1 or len_of_data < self.minimum_data_points or depth<self.max_Depth or node_purity >= 0.95:
            stop_splits = True
        return stop_splits
    
        
        
        
    
    def fit(self, df, attributes): # Will have to see if I have to pass self.index, self.model_params, self.level as well (that will happen if class variables aren't able to hold values across recurssive function calls)
        '''
        This is the main training function which trains the DT.
        It creates and implements the primary data structure for housing the DT mdoel.
        It checks (perhaps with another fucntion), to decide if the splitting criteria has been met, and when to stop going deeper
        It calls the attribute_selector(), at each iteration of data. It will also pass the list of attributes to select from. 
        It recurssively calls itself as it goes deeper and deeper into the tree (this step is optional, and lets see if we can perform this with iteration)
        
        Args:
            df (DataFrame) -> Containing the training records (updated at every recurring step)
            attributes (iterable) -> List containing the remaining attributes
        
        
        '''
        self.level[0] = self.level[0] + 1
        stopping_condition = self.check_split_criteria(df) # Will probably have to pass quite a few details here as arguments, will see later...     
        column_names = ['attribute_name', 'level', 'attribute_type', 'condition','child_index', 'parent_index', 'class_label']
        temp_param = pd.DataFrame(columns = column_names)
        temp_param['child_index'] = temp_param['child_index'].astype('object')
        temp_param['child_index'] = [0, 0] # This would be an array of length equal to the number of splits in C4.5 implementation
        if stopping_condition:
            temp_param = pd.DataFrame(columns = column_names)
            # Code for assigning class labels
            temp_param.class_label = self.node_class_label(df)
            temp_param.level = self.level[0]
            temp_param.attribute_type = 'Leaf'
            temp_param.parent_index = self.index[0]
            
            # I will conclude the class label and add it to the self.model_params
            self.model_params = pd.concat([self.model_params, temp_param], ignore_index = True)
            self.leve[0] = self.level[0] - 1
            self.index[0] = self.index[0] + 1
            return self.level, self.index
        else:
            temp_param.condition, temp_param.attribute_name = self.attribute_selector(df, self.attribute_list)
            temp_param.level = self.level[0]
            temp_param.attribute_type = 'Root'
            temp_param.parent_index = self.index[0]
            self.index[0] = self.index[0] + 1
            self.model_params = pd.concat([self.model_params, temp_param], ignore_index = True)
            self.attribute_list = self.attribute_list.drop(temp_param.attribute_name)
            condition_array = np.array([np.NINF, temp_param.condition, np.INF])
            attribute_name = temp_param.attribute_name
            current_index = len(self.model_params) - 1 # This could also be self.index[0]. This would only work if python stores a variable stack of the caller function during the recurssive call
            for i in range(2):
                df_updated = df.loc[(df[attribute_name]> condition_array[i]) & (df[attribute_name] <= condition_array[i+1])]
                self.level[0], child_index_value = self.fit(df = df_updated, attributes = self.attribute_list)
                self.model_params.loc[current_index, 'child_index'][i] = child_index_value
            self.attribute_list.append(temp_param.attribute_name) # This too, depends on whether python stores the variable stack of the caller function
            self.level[0] = self.level[0] - 1
            return self.level, current_index          
        
              
        
        
        
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
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    