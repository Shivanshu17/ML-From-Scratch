def standard_scaler_LSTM(X_train, X_test):
    X_train_scaled = []
    X_test_scaled = []
    scaler = StandardScaler()
    print("Shape of X_train is", X_train.shape)
    print("Shape of X_test is", X_test.shape)
    for i in range(X_train.shape[0]):
        X_train_scaled.append(scaler.fit_transform(X_train[i]).tolist())
    for j in range(X_test.shape[0]):
        X_test_scaled.append(scaler.transform(X_test[j]).tolist())
    #display(X_train_scaled)
    #display(X_test_scaled.shape)
    return np.array(X_train_scaled), np.array(X_test_scaled)



def multivariate_data(dataset, target, end_index, start_index = 0, history_size = 50, target_size = 180, step = 1, single_step=False, seasonal_lag = False, slen = 365, seasonal_history_size = 30):
    '''
    This function converts the given data into batches that can be fed into the LSTM model
    
    
    '''
    data = []
    data1 = []
    label_values = []
    if seasonal_lag:
        start_index = start_index + slen + seasonal_history_size
    else:
        start_index = start_index + history_size
        
    print("Start Index is ", start_index)
    
    if end_index is None:
        end_index = dataset.shape[0] - target_size
    
    print("End index is ", end_index)

    for i in range(start_index, end_index):
        
        if seasonal_lag:
            indices = range(i-slen-seasonal_history_size, i-slen, step)
            data.append(dataset[indices])
        else:
            indices = range(i-history_size, i, step)
            data.append(dataset[indices])
            
        if single_step:
            label_values.append(target[i+target_size])
        else:
            label_values.append(target[i:i+target_size])
    #print("Shape of label values is ", label_values.shape)
    return np.array(data), np.array(label_values)


def create_data_for_LSTM(dataset, history_size, test_size = 180, seasonal_lag = False, seasonal_history_size = 30, slen = 365):
    '''
    This function divides the data into test and train parts by calling the multvariate_data function. It then forms
    
    
    
    '''
    STEP = 1
    BATCH_SIZE = 50
    if seasonal_lag:
        test_start_point = dataset.shape[0] - 2*test_size - slen - seasonal_history_size
    else:
        test_start_point = dataset.shape[0] - 2*test_size - history_size
    TRAIN_SPLIT = dataset.shape[0] - test_size 
    #display(TRAIN_SPLIT)
    #display(test_start_point)
    #display(dataset[:,0].shape)
    print("Following part happens for train data generation")
    x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], start_index = 0, end_index = TRAIN_SPLIT, history_size = history_size, seasonal_lag = seasonal_lag,seasonal_history_size = seasonal_history_size, target_size = test_size, step = STEP)
    print("Following part happens for test data generation")
    x_test_multi, y_test_multi = multivariate_data(dataset, dataset[:, 0], start_index = test_start_point, end_index = None, history_size = history_size, seasonal_lag = seasonal_lag, seasonal_history_size = seasonal_history_size, target_size = test_size, step = STEP)
    print ('Single window of past history : {}'.format(x_train_multi[0].shape))
    print('Shape of x_train_multi is ', x_train_multi.shape)
    print('Shape of x_test_multi is ', x_test_multi.shape)
    print('Shape of y_train_multi is ', y_train_multi.shape)
    print('Shape of y_test_multi is ', y_test_multi.shape)
    
    x_train_multi_scaled, x_test_multi_scaled = standard_scaler_LSTM(x_train_multi, x_test_multi)
    
    train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi_scaled, y_train_multi))
    #train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_data_multi = train_data_multi.cache().batch(BATCH_SIZE).repeat()

    test_data_multi = tf.data.Dataset.from_tensor_slices((x_test_multi_scaled, y_test_multi))
    #test_data_multi = test_data_multi.batch(BATCH_SIZE).repeat()
    test_data_multi = test_data_multi.batch(BATCH_SIZE).repeat()
    
    return x_train_multi_scaled, train_data_multi, test_data_multi



def LSTM_model_1(dataset_type = 1, test_size = 180, plot_intervals = True, plot_anomalies = False, scale = 1.25):
    '''
    This function defines the sequential LSTM model and implements it on the data defined by the "dataset_type" passed to this function
    
    
    '''
    
    if dataset_type == 1:
        data_df1 = store_item_sale_TS_1_1.copy()
        data_df = store_item_sale_TS_1_1.copy()
        data_df1 = data_df1.dropna()
        dataset = data_df1.values
        x_train_multi_scaled, train_data_multi, test_data_multi = create_data_for_LSTM(dataset, history_size = 20, test_size = 180, seasonal_lag = False, seasonal_history_size = 30)
        
    if dataset_type == 2:
        data_df1 = store_item_sale_TS_1_1.copy()
        data_df = store_item_sale_TS_1_1.copy()
        data_df1 = data_df1.dropna()
        dataset = data_df1.values
        x_train_multi_scaled, train_data_multi, test_data_multi = create_data_for_LSTM(dataset, history_size = 30, test_size = 180, seasonal_lag = True, seasonal_history_size = 50)
        
    if dataset_type == 3:
        data_df1 = store_item_sale_TS_2_1.copy()
        data_df = store_item_sale_TS_2_1.copy()
        data_df1 = data_df1.dropna()
        dataset = data_df1.values
        x_train_multi_scaled, train_data_multi, test_data_multi = create_data_for_LSTM(dataset, history_size = 2, test_size = 180, seasonal_lag = False, seasonal_history_size = 30)
        
    if dataset_type == 4:
        data_df1 = store_item_sale_TS_2_2.copy()
        data_df = store_item_sale_TS_2_2.copy()
        data_df1 = data_df1.dropna()
        dataset = data_df1.values
        x_train_multi_scaled, train_data_multi, test_data_multi = create_data_for_LSTM(dataset, history_size = 5, test_size = 180, seasonal_lag = False, seasonal_history_size = 30)
        
    if dataset_type == 5:
        data_df1 = store_item_sale_TS_3_1.copy()
        data_df = store_item_sale_TS_3_1.copy()
        data_df1 = data_df1.dropna()
        dataset = data_df1.values
        x_train_multi_scaled, train_data_multi, test_data_multi = create_data_for_LSTM(dataset, history_size = 30, test_size = 180, seasonal_lag = True, seasonal_history_size = 30) # This part is to be decided after evaluating the results derived from above iterations
        
    EPOCHS = 20
    model_1 = tf.keras.models.Sequential()
    model_1.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=x_train_multi_scaled.shape[-2:]))
    model_1.add(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))
    model_1.add(tf.keras.layers.Dropout(0.3))
    model_1.add(tf.keras.layers.LSTM(16, activation='relu'))
    #model_1.add(tf.keras.layers.Dropout(0.3))
    model_1.add(tf.keras.layers.Dense(180))

    model_1.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mse')
    
    for x, y in test_data_multi.take(1):
        print (model_1.predict(x).shape)
        
    model_1_history = model_1.fit(train_data_multi, epochs=EPOCHS, validation_data=test_data_multi, validation_steps=50, steps_per_epoch = 100)
    
    
    
    for x, y in test_data_multi.take(1):
        prediction = model_1.predict(x)[0]
    
    
    
    # Now we move to the plotting part
    
    predicted_df = pd.DataFrame(prediction, columns = ['sales'])
    predicted_df.index = store_item_sale_TS_test.index 
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
        l1 = len(agg_predicted)
        
        anomalies_aggregated = np.array([np.NaN]*l1)
        anomalies_aggregated = pd.DataFrame(anomalies_aggregated)
        anomalies_aggregated.index = agg_predicted.index
        drop_list1 = []
        for j in range(len(anomalies_aggregated)):
            if agg_actual.values[j][0] < LowerBound_aggregated.values[j][0]:
                anomalies_aggregated.iloc[j] = agg_actual.values[j][0]
            if agg_actual.values[j][0] > UpperBound_aggregated.values[j][0]:
                anomalies_aggregated.iloc[j] = agg_actual.values[j][0]
            else:
                drop_list1.append(j)
        anomalies_aggregated.drop(anomalies_aggregated.index[drop_list1], axis = 0, inplace = True)
        
        
        plt.plot(UpperBound_aggregated['sales'], "r--", alpha = 0.7, label = "Upper Bound/ Lower Bound")
        plt.plot(LowerBound_aggregated['sales'], "r--", alpha = 0.7)
        plt.plot(anomalies_aggregated, "o", color = "r", markersize = 5, label = "Anomalies")
        plt.fill_between(x = agg_predicted.index, y1 = UpperBound_aggregated.values.ravel(), y2 = LowerBound_aggregated.values.ravel(), alpha = 0.2, color = "grey")
        plt.axvspan(predicted_df.index[0], predicted_df.index[-1], alpha = 0.4, color = "grey")
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.show()
    
    return model_1


