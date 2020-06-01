class SARIMA:
    '''
    This class calculates the best parameter value, forecasts future values on the derived set of parameters and plots the results
    
    Args:
        initial_values (iterable) -> List representing the range of the parameters in order - ps, qs, Ps, Qs (two values for each)
        series (DataFrame) -> Train timeseries of sales
        n_preds (integer) -> The number of predictions
        D (integer) -> Seasonality difference parameter
        d (integer) -> Trend difference index
        slen (integer) -> Length of the season
    
    '''
    def __init__(self, series, initial_values, n_preds = 180, D = 1, d = 7, slen = 365):
        self.series = series
        self.initial_values = initial_values
        self.n_preds = n_preds
        self.d = d
        self.D = D
        self.slen = slen
        
        
    def tune_SARIMA(self):
        '''
        This function finds the best combination of parameters for SARIMA model
        
        Returns:
            result_table (dataframe) -> The combination of Parameters along with their respective AIC values as calculated by SARIMAX
        
        '''
        self.result_table = []
        ps = range(self.initial_values[0], self.initial_values[1])
        qs = range(self.initial_values[2], self.initial_values[3])
        Ps = range(self.initial_values[4], self.initial_values[5])
        Qs = range(self.initial_values[6], self.initial_values[7])
        parameters_list = list(product(ps,qs,Ps,Qs))        
        best_aic = float("inf")
        for param in tqdm_notebook(parameters_list):
            # we need try-except because on some combinations model fails to converge
            try:
                model=sm.tsa.statespace.SARIMAX(self.series.sales, order=(param[0], self.d, param[1]), 
                                                seasonal_order=(param[2], self.D, param[3], self.slen)).fit(disp=-1)
            except:
                continue
            aic = model.aic
            # saving best model, AIC and parameters
            if aic < best_aic:
                best_model = model
                best_aic = aic
                best_param = param
            results.append([param, model.aic])

        self.result_table = pd.DataFrame(results)
        self.result_table.columns = ['parameters', 'aic']
        # sorting in ascending order, the lower AIC is - the better
        self.result_table = self.result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
        p,q,P,Q = self.result_table.parameters[0]
        print('The best parameter values are : p - ', p, " q - ", q, 'P - ', P, ' Q - ',Q)
        
    
    def fit_SARIMA(self, test_data):
        '''
        This function fits the parameters and makes a forecast
        Also, this function can be used to determine the anomalies as well. I will havet to figure out how that can 
        be done in SARIMA modeling, but that shouldn't be too tough.
        
        Args:
            test_data (DataFrame) -> test part of the sales timeseries
        
        '''
        self.tune_SARIMA()
        self.forecast = []
        p, q, P, Q = self.result_table.parameters[0]
        best_model=sm.tsa.statespace.SARIMAX(self.series.sales, order=(p, self.d, q), seasonal_order=(P, self.D, Q, self.slen)).fit(disp=-1)
        print(best_model.summary())
        a_df = pd.concat(self.series, test_data, join = 'inner', axis = 0)
        forecast = best_model.predict(start = (self.slen+self.d), end = self.series.shape[0]+self.n_preds)
        self.forecast = np.array((np.NaN)*(self.slen + self.d))
        self.forecast = self.forecast.append(forecast)
        self.forecast = pd.DataFrame(self.forecast, columns = 'sales')
        self.forecast.index = pd.to_datetime(a_df.index)
        
        
    def plot_SARIMA(self, actual_values, aggregated = True):
        '''
        This function plots the results derived from SARIMA modelling. I still have to add the code for Anomalies.
        
        Args:
            actual_values (DataFrame) -> Timeseries representing sales_df, with test set included.
            aggregated (Boolean) -> Determines if we want to print the aggregated scores of values
        
        '''
        error_pred = mean_squared_error(actual_values['sales'][0-self.n_preds:], self.forecast['sales'][0-self.n_preds:])
        plt.figure(figsize = (15,7))
        plt.title("Mean Squared Error: {0:.2f}%".format(error_pred))
        if aggregated:
            agg_forecast = self.forecast.resample('W', label = 'left').sum() # Have to see if I should handle NaN values with fillna
            agg_actual_values = actual_values.resample('W', label = 'left').sum()
            plt.plot(agg_forecast, color = "g", label = "Predicted Values")
            plt.plot(agg_actual_values, color = 'b', label = "Actual Values")
        else:
            plt.plot(self.forecast, color = "g", label = "Predicted Values")
            plt.plot(actual_values, color = "b", label = "Actual Values")
        plt.axvspan(self.forecast.index[0-self.n_preds], self.forecast.index[-1], alpha = 0.4, color = 'grey')
        plt.legend(loc = 'upper left')
        plt.grid(True)
        plt.show()
        
        
        
        
        
        
        