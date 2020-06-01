class Double_ES:
    
    """
    This class performs double exponential smoothing and updates the deviation, upper bound, and lower bound values for each time step. It also plots the results with its own function.
    There is no limit to the prediction horizon, however, since we are only relying on the trend update to make the predictions, it wouldn't be wise to set the n_preds value to be too 
    large.
    
    Args:
        series (iterable) -> Actual values array
        alpha, beta (float) -> Double exponential smoothing parameters
        n_preds (int) -> predictions horizon
        scaling_factor (float) - sets the width of the confidence interval and is used to determine the upper bound and lower bound
    
    """
    
    
    def __init__(self, series, alpha, beta, n_preds, scaling_factor=1):
        if type(series) is np.ndarray:
            series = pd.DataFrame(series)
        self.series = series
        self.alpha = alpha
        self.beta = beta
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(len(self.series)-1):
            sum += float(self.series.iloc[i+1] - self.series.iloc[i])
        return sum /(len(self.series))

          
    def double_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBound = []
        self.LowerBound = []
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series.iloc[0]
                trend = self.initial_trend()
                self.result.append(self.series.iloc[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                
                self.PredictedDeviation.append(0)
                
                # Can't decide if I should add the error term (as I did in plot_results() function), into the evaluation of upper and lower bounds. I will have to rewrite that 
                # entire function here, and take metric as an argument as well. Will have to see the plot, to figure out if I should.
                self.UpperBound.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBound.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                
                # We could apply smoothing to the 'm*trend' part of the code in the next line, it will ensure that the function isn't running just on final trend value, but is 
                # also considering the trend values of the earlier time steps. Will have to see the results to alter this part.
                self.result.append(smooth + m*trend)
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series.iloc[i]
                last_smooth, smooth = smooth, self.alpha*(val) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                self.result.append(smooth+trend)
                
                # Instead of calculating the standard deviation of the results, we will here calculate the deviation using a variant of brutlag method, with the only difference being,
                # instead of using gamma as the parameter for deviation updata, we will use a fixed value of 0.5 (which we can adjust according to the results)
                self.PredictedDeviation.append(0.5 * np.abs(self.series.iloc[i] - self.result[i]) 
                                               + (1- 0.5)*self.PredictedDeviation[-1])
            
            # Again, the error term can be added here (to grant the model more versatility)
            self.UpperBound.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBound.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)

        
    def plot_Double_ES(self, test_data, plot_intervals = True, plot_anomalies = True, aggregated = True):
        '''
        Function plots the results for DES. It can plot the results, lower bound, and upper bound even for time steps that do not have actual values given.
        This plot function is more geared towards the actual industry situations where we are deploying the model for predictions without having the 'actual observed values' given.
        
        args:
            test_data (dataframe) -> Dataframe representing the last 180 values indexed according to their date time index values
            plot_intervals (boolean) -> Defines if we want to plot the confidence intervals
            plot_anomalies (boolean) -> Defines if we want to plot the anomalies
            aggregated (boolean) -> Determines whether or not we want to resample our plots on a weekly basis
        
        '''
        self.result = pd.DataFrame(np.array(self.result))
        any_df = pd.concat([self.series, test_data], axis = 0)
        self.result.index = pd.to_datetime(any_df.index)
        t = self.series.index
        l = len(self.series)
        self.series = pd.concat([self.series, test_data], axis = 0)
        
        if plot_intervals:
            self.UpperBound = pd.DataFrame(self.UpperBound)
            self.UpperBound.index = self.result.index
            self.LowerBound = pd.DataFrame(self.LowerBound)
            self.LowerBound.index = self.result.index
            
        if plot_anomalies:
            self.anomalies = np.array([np.NaN]*l)
            self.anomalies = pd.DataFrame(self.anomalies)
            self.anomalies.index = t
            drop_list = []
            
            for i in range(len(self.anomalies)):
                if self.series.values[i][0] < self.LowerBound.values[i][0]:
                    self.anomalies.iloc[i] = self.series.values[i][0]
                if self.series.values[i][0] > self.UpperBound.values[i][0]:
                    self.anomalies.iloc[i] = self.series.values[i][0]
                else:
                    drop_list.append(i)
            
            self.anomalies.drop(self.anomalies.index[drop_list], axis = 0, inplace = True)
        
        # This part prints the results in an aggregated form and everything is resampled according to weekly basis
        if aggregated == True:
            results_aggregated = self.result.resample('W', label = 'left').sum()
            actual_aggregated = self.series.resample('W', label = 'left').sum()
            UpperBound_aggregated = self.UpperBound.resample('W', label = 'left').sum()
            LowerBound_aggregated = self.LowerBound.resample('W', label = 'left').sum()
            
            # The following code is to find aggregated anomalies. 
            l1 = len(results_aggregated)
            anomalies_aggregated = np.array([np.NaN]*l1)
            anomalies_aggregated = pd.DataFrame(anomalies_aggregated)
            anomalies_aggregated.index = results_aggregated.index
            drop_list1 = []
            for j in range(len(anomalies_aggregated)):
                if actual_aggregated.values[j][0] < LowerBound_aggregated.values[j][0]:
                    anomalies_aggregated.iloc[j] = actual_aggregated.values[j][0]
                if actual_aggregated.values[j][0] > UpperBound_aggregated.values[j][0]:
                    anomalies_aggregated.iloc[j] = actual_aggregated.values[j][0]
                else:
                    drop_list1.append(j)
            anomalies_aggregated.drop(anomalies_aggregated.index[drop_list1], axis = 0, inplace = True)

            plt.figure(figsize = (15,7))
            plt.plot(results_aggregated, "g", label ="Predicted Values")
            plt.plot(actual_aggregated, "b", label = "Actual Values")
            plt.plot(UpperBound_aggregated, "r--", alpha = 0.3, label = "Upper Bound/ Lower Bound")
            plt.plot(LowerBound_aggregated, "r--")
            plt.fill_between(x = results_aggregated.index, y1 = UpperBound_aggregated.values.ravel(), y2 = LowerBound_aggregated.values.ravel(), alpha = 0.2, color = "grey")
            plt.axvspan(test_data.index[0], test_data.index[-1], alpha = 0.4, color = "grey")
            plt.plot(anomalies_aggregated, "o", color = "r", markersize = 5, label = "Anomalies")
            plt.grid(True)
            plt.legend(loc = "upper left")
            plt.show()
        
        # The following part prints the results without aggregation
        else:
            plt.figure(figsize = (15,7))
            plt.plot(self.result, "g", label = "Predicted Values")
            plt.title("Prediction values for sales using Triple Exponential Smoothing")
            plt.plot(self.series, "b", label = "Actual Values")
            plt.plot(self.UpperBound, "r--", alpha = 0.3, label = "Upper Bound/ Lower Bound")
            plt.plot(self.LowerBound, "r--", alpha = 0.3)
            plt.fill_between(x = self.result.index, y1 = self.UpperBound.values.ravel(), y2 = self.LowerBound.values.ravel(), alpha = 0.2, color = "grey")
            plt.axvspan(test_data.index[0], test_data.index[-1], alpha = 0.6, color = 'grey')
            plt.plot(self.anomalies, "o", color = "r", markersize = 5, label = "Anomalies")
            plt.show()       
        