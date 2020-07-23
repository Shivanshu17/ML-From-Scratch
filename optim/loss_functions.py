from functools import reduce
import math
import pandas as pd
def mean_squared_loss(y, pred_y):
    assert len(y) == len(pred_y)
    return (reduce((lambda item1, item2: item1 + item2), map(lambda x: x**2, (y-pred_y)))/len(pred_y))

def mean_absolute_loss(y, pred_y):
    assert len(y) == len(pred_y)
    return (reduce((lambda item1, item2: item1 + item2), map(abs, (y - pred_y)))/len(pred_y))

def huber_loss(y, pred_y, t):
    assert len(y) == len(pred_y)
    summed_up = sum(map(abs, y-pred_y))
    if summed_up < t:
        return (reduce((lambda item1, item2: item1 + item2), map(lambda x:x**2, (y-pred_y)))/len(pred_y))
    else:
        return( t*reduce((lambda item1, item2: item1 + item2), map(abs, (y-pred_y)))/len(pred_y) - (t**2)/2)
    
def log_cosh_loss(y, pred_y):
    assert len(y) == len(pred_y)
    log_cosh = lambda x: math.log(math.cosh(x))
    return (reduce((lambda item1, item2: item1 + item2), map(log_cosh, map(abs, y-pred_y))))

def quantile_loss(y, pred_y, q):
    assert len(y) == len(pred_y)
    assert q<=1 and q>=0
    loss = 0
    for i in range(len(y)):
        if (y[i]-pred_y[i]>0):
            loss = loss + q*(y[i]-pred_y[i])
        else:
            loss = loss + (q-1)*(y[i] - pred_y[i])
    
    return (loss/len(y))



# Classification Errors
    
def gini_loss(self, df, condition_array, type_of_data = 'categorical'):
    '''
    This function calculates the impurity percentage of the attribute split with gini_index
    
    Args:
        df (Series) -> pd.Series containing a single attribute column of the original data
        condition_array (iterable) -> An array containing either the categories of the left split, or a single numerical value to measure continuous data against.
        type_of_data (object) -> Can either be 'categorical' or 'continuous'
    
    Returns:
        A gini_index score for that node with the given condition
    
    '''
    _TYPE = ('categorical','continuous')
    assert type_of_data is in _TYPE, 'type_of_data can only be either categorical or continuous'
    if type == 'categorical':
        count = len(df.isin(condition_array))
    if type == 'continuous':
        count = len(df[df < condition_array[0]]) # Have to make sure that this line is correct
    p = count/df.shape[0]           
    return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))
    
    
    
    

def entropy_loss(self, df, condition_array, type_of_data = 'categorical'):
    '''
    This function calculates the impurity percentage of the attribute split with entropy_loss
    
    Args:
        df (Series) -> pd.Series containing a single attribute column of the original data
        condition_array (iterable) -> An array containing either the categories of the left split, or a single numerical value to measure continuous data against.
        type_of_data (object) -> Can either be 'categorical' or 'continuous'
    
     Returns:
        A gini_index score for that node with the given condition
    
    '''
    _TYPE = ('categorical','continuous')
    assert (type_of_data is in _TYPE):
        print('type_of_data can only be either categorical or continuous')
    if type == 'categorical':
        count = len(df.isin(condition_array))
    if type == 'continuous':
        count = len(df[df < condition_array[0]]) # Have to make sure that this line is correct
    p = count/df.shape[0]           
        
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))
    
    
    
    
def classification_error(self,  df, condition_array, type_of_data = 'categorical'):
    '''
    This function calculates the impurity metric of the attribute using simple classification error
    
    Args:
        df (Series) -> pd.Series containing a single attribute column of the original data
        condition_array (iterable) -> An array containing either the categories of the left split, or a single numerical value to measure continuous data against.
        type_of_data (object) -> Can either be 'categorical' or 'continuous'
    
    Returns:
        A gini_index score for that node with the given condition
    
    '''
    _TYPE = ('categorical','continuous')
    assert (type_of_data is in _TYPE):
        print('type_of_data can only be either categorical or continuous')
    if type == 'categorical':
        count = len(df.isin(condition_array))
    if type == 'continuous':
        count = len(df[df < condition_array[0]]) # Have to make sure that this line is correct
    p = count/df.shape[0]           
    return 1 - np.max([p, 1 - p])
    


    
