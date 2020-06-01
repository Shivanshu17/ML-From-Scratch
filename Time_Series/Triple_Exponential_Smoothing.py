class Triple_ES:
    
    """
    In Triple Exponential Smoothing, we implement the Holt-Winters model, and derive the outliers using the brutlag method
    
    Args:
        series (iterbale) -> Actual values array
        slen (int) -> Length of a season
        alpha, beta, gamma (float) -> Holt-Winters model coefficients
        n_preds (int) -> predictions horizon
        scaling_factor (float) -> sets the width of the confidence interval by Brutlag (usually takes values from 2 to 3)
    
    """
    
    
    def __init__(self, series, alpha, beta, gamma, n_preds = 180, scaling_factor=1.5, slen = 365):
        if type(series) is np.ndarray:
            series = pd.DataFrame(series)
        self.series = series
        self.slen = int(slen)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor
        
        
    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series.iloc[i+self.slen] - self.series.iloc[i]) / self.slen
        return sum / self.slen  
    
    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
        # let's calculate season averages
        for j in range(n_seasons):
            season_averages.append((self.series.iloc[self.slen*j:self.slen*j+self.slen].sum()[0])/float(self.slen))
        # let's calculate initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series.iloc[self.slen*j+i]-season_averages[j]
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals   

          
    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBound = []
        self.LowerBound = []
        
        seasonals = self.initial_seasonal_components()
        
        for i in range(len(self.series)+self.n_preds):
            if i == 0: # components initialization
                smooth = self.series.iloc[0]
                trend = self.initial_trend()
                self.result.append(self.series.iloc[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i%self.slen])
                
                self.PredictedDeviation.append(0)
                
                self.UpperBound.append(self.result[0] + 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                
                self.LowerBound.append(self.result[0] - 
                                      self.scaling_factor * 
                                      self.PredictedDeviation[0])
                continue
                
            if i >= len(self.series): # predicting
                m = i - len(self.series) + 1
                self.result.append((smooth + m*trend) + seasonals[i%self.slen])
                
                # when predicting we increase uncertainty on each step
                self.PredictedDeviation.append(self.PredictedDeviation[-1]*1.01) 
                
            else:
                val = self.series.iloc[i]
                last_smooth, smooth = smooth, self.alpha*(val-seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                trend = self.beta * (smooth-last_smooth) + (1-self.beta)*trend
                seasonals[i%self.slen] = self.gamma*(val-smooth) + (1-self.gamma)*seasonals[i%self.slen]
                self.result.append(smooth+trend+seasonals[i%self.slen])
                
                # Deviation is calculated according to Brutlag algorithm.
                self.PredictedDeviation.append(self.gamma * np.abs(self.series.iloc[i] - self.result[i]) 
                                               + (1-self.gamma)*self.PredictedDeviation[-1])
                     
            self.UpperBound.append(self.result[-1] + 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.LowerBound.append(self.result[-1] - 
                                  self.scaling_factor * 
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i%self.slen])

    
    
    def plot_Triple_ES(self, test_data, plot_intervals = True, plot_anomalies = True, aggregated = True):
        '''
        Function plots the results for TES. It can plot the results, lower bound, and upper bound even for time steps that do not have actual values given.
        This plot function is more geared towards the actual industry situations where we are deploying the model for predictions without having the 'actual observed values' given.
        
        args:
            test_data (dataframe) -> Datetime index wise representation of the last 180 day values.
            plot_intervals (boolean) -> Defines if we want to plot the confidence intervals
            plot_anomalies (boolean) -> Defines if we want to plot the anomalies
            aggregated (boolena) -> Decides whether or not we want to resample our output on a weekly basis
        
        '''
        self.result = pd.DataFrame(self.result)
        any_df = pd.concat([self.series, test_data], join = "inner", axis = 0)
        self.result.index = pd.to_datetime(any_df.index)
        t = self.series.index
        l = len(self.series)
        self.series = pd.concat([self.series, test_data], axis = 0)
        #display(self.series.tail(20))
        #display(test_data.tail(20))
        #display(self.result)
        
        if plot_intervals:
            self.UpperBound = pd.DataFrame(self.UpperBound)
            self.UpperBound.index = self.result.index
            self.LowerBound = pd.DataFrame(self.LowerBound)
            self.LowerBound.index = self.result.index

        if plot_anomalies:
            self.anomalies = np.array([np.NaN]*(l))
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