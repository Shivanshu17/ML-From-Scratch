def XGBoost_model(info_level = 2, test_size = 180, scaling_metric = 1, plot_intervals = True, plot_anomalies = False, scale = 1.25):
    '''
    This function trains the Linear Regression model on whichever level of data it is assigned and plots the results.
    
    '''
    
    if info_level == 1:
        data_df = store_item_sale_TS_1.copy()
    if info_level == 2:
        data_df = store_item_sale_TS_2.copy()
    if info_level == 3:
        data_df = store_item_sale_TS_3.copy()
    
    # First we get the data in the required format
    data_df = data_df.dropna()
    data_df_y = data_df['sales']
    data_df_X = data_df.drop(['sales'], axis = 1)
    X_train, X_test, y_train, y_test = timeseries_train_test_split(data_df_X, data_df_y, test_size = test_size)
    
    # Then we scale the data
    if scaling_metric == 1:
        X_train_scaled, X_test_scaled = standard_scaler(X_train, X_test)
    if scaling_metric == 2:
        X_train_scaled, X_test_scaled = minmax_scaler(X_train, X_test)
    
    # We now initialize the model and train it
    xgb = XGBRegressor()
    xgb.fit(X_train_scaled, y_train)
    prediction = xgb.predict(X_test_scaled)
    #display(prediction)
    
    
    # We are now gonna plot the results along with coefficient scores
    
    predicted_df = pd.DataFrame(prediction, columns = ['sales'])
    predicted_df.index = X_test.index
    # Total error
    error_pred = float(mean_squared_error(data_df['sales'][0-test_size:],predicted_df['sales']))
    
    plt.figure(figsize = (15,7))
    plt.title("Mean Squared Error {0:.2f}".format(error_pred))
    agg_predicted = predicted_df.resample('W', label = 'left').sum()
    agg_actual = data_df.resample('W', label = 'left').sum()
    plt.plot(agg_predicted, color = "g", label = "Predicted Values")
    plt.plot(agg_actual['sales'], color = "b", label = "Actual Values")
    
    if plot_intervals:
        deviation = []
        lower_bound = []
        upper_bound = []
        agg_actual_reindexed = agg_actual.set_index(pd.Index(range(0,len(agg_actual))))
        agg_predicted_reindexed = agg_predicted.set_index(pd.Index(range(0,len(agg_predicted))))
        deviation = np.std((agg_actual_reindexed.iloc[(0-len(agg_predicted)):, 0].values, agg_predicted_reindexed['sales'].values)) 
        lower_bound = np.array(agg_predicted['sales'].values) - ((error_pred/10) + scale*deviation)
        upper_bound = np.array(agg_predicted['sales'].values) + ((error_pred/10) + scale*deviation)
        LowerBound_aggregated = pd.DataFrame(lower_bound, columns = ['sales'])
        LowerBound_aggregated.index = agg_predicted.index
        UpperBound_aggregated = pd.DataFrame(upper_bound, columns = ['sales'])
        UpperBound_aggregated.index = agg_predicted.index
        plt.plot(UpperBound_aggregated['sales'], "r--", alpha = 0.3, label = "Upper Bound/ Lower Bound")
        plt.plot(LowerBound_aggregated['sales'], "r--")
        plt.fill_between(x = agg_predicted.index, y1 = UpperBound_aggregated.values.ravel(), y2 = LowerBound_aggregated.values.ravel(), alpha = 0.2, color = "grey")
        plt.axvspan(X_test.index[0], X_test.index[-1], alpha = 0.4, color = "grey")
        
    # I still have to write the code for anomalies
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.show()
    
    return xgb
