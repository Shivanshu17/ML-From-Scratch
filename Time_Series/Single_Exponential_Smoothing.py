class Single_ES:
    
    """
    This model object will produce the Single Exponential smoothing values. Since it does not have the abilities to either smooth or to add trend, it can only be used to make a single 
    meaningful prediction. That is, just one time step can be predicted.
    Args:
        series (iterable) -> Actual time series
        alpha (float) -> Represents the parameter for single exponential smoothing
        scaling_factor (float) -> sets the width of the confidence interval
        
    Returns:
        result (iterable) -> A numpy array containing the results of smoothing
    
    """
    
    
    def __init__(self, series, alpha = 0.9, scaling_factor=2):
        self.series = series
        self.alpha = alpha
        # Scaling factor will only be used, if the plot is done here in this class
        self.scaling_factor = scaling_factor
        
          
    def single_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.result = [series[0]] # first value is same as series
        for t in range(1, len(series)):
            self.result.append(self.alpha * self.series[t] + (1 - self.alpha) * self.result[t-1])
        # Since we can't make more than one prediction, we can either print the plot here, or we can print the plot from the calling function where the 'model' object is created. 
        # There's no difference in the plots drawn from either.
        return np.array(self.result)