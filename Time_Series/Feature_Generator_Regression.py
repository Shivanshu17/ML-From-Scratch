store_item_sale_TS = create_data(storewise_train, store_id, item_id)
store_item_sale_TS_train= store_item_sale_TS.iloc[:-180]
store_item_sale_TS_test = store_item_sale_TS.iloc[-180:]
store_item_sale_df_not_TS = store_item_sale_TS.reset_index()

store_item_sale_TS_0 = store_item_sale_TS.copy()
store_item_sale_TS_0['day_of_year'] = 0
store_item_sale_TS_0['week_of_year'] = 0
store_item_sale_TS_0['month_of_year'] = 0
store_item_sale_TS_0['day_of_month'] = 0
store_item_sale_TS_0['day_of_week'] = 0
store_item_sale_TS_0['weekend'] = 0
store_item_sale_TS_0['holiday'] = 0

for date in store_item_sale_TS_0.index:
    store_item_sale_TS_0.loc[date, 'day_of_year'] = date.dayofyear
    store_item_sale_TS_0.loc[date, 'week_of_year'] = date.weekofyear
    store_item_sale_TS_0.loc[date, 'month_of_year'] = date.month
    store_item_sale_TS_0.loc[date, 'day_of_month'] = date.day
    store_item_sale_TS_0.loc[date, 'day_of_week'] = date.dayofweek
    if date in us_holidays:
        store_item_sale_TS_0.loc[date,'holiday'] = 1
    if date.dayofweek == 6 or date.dayofweek == 5:
        store_item_sale_TS_0.loc[date,'weekend'] = 1
    
display(store_item_sale_TS_0.head(10))

# Adding lag features
store_item_sale_TS_1 = store_item_sale_TS_0.copy()
for i in range(365,365+28):
    store_item_sale_TS_1["lag_{}".format(i)] = store_item_sale_TS_1.sales.shift(i)
display(store_item_sale_TS_1.head(10))
display(store_item_sale_TS_1.tail(10))

# Adding window features
month_lag_colname_list = ["lag_{}".format(i + 365) for i in range(0,28)]
week1_lag_colname_list = ["lag_{}".format(i + 365) for i in range(0,7)]
week2_lag_colname_list = ["lag_{}".format(i + 365) for i in range(7,14)]
week3_lag_colname_list = ["lag_{}".format(i + 365) for i in range(14,21)]
week4_lag_colname_list = ["lag_{}".format(i + 365) for i in range(21,28)]
fortnight_lag_colname_list = ["lag_{}".format(i + 365) for i in range(0,14)]

store_item_sale_TS_2 = store_item_sale_TS_1.copy()

store_item_sale_TS_2['month_lag_mean'] = 0
store_item_sale_TS_2['month_lag_max'] = 0
store_item_sale_TS_2['month_lag_min'] = 0
store_item_sale_TS_2['month_lag_variance'] = 0

store_item_sale_TS_2['week1_lag_mean'] = 0
store_item_sale_TS_2['week1_lag_max'] = 0
store_item_sale_TS_2['week1_lag_min'] = 0
store_item_sale_TS_2['week1_lag_variance'] = 0

# display(month_lag_colname_list)

for date in store_item_sale_TS_2.index:
    month_lag_series = store_item_sale_TS_2.loc[date, month_lag_colname_list]
    week1_lag_series = store_item_sale_TS_2.loc[date, week1_lag_colname_list]
    
    store_item_sale_TS_2.loc[date, 'month_lag_mean'] = month_lag_series.mean()
    store_item_sale_TS_2.loc[date, 'month_lag_max'] = month_lag_series.max()
    store_item_sale_TS_2.loc[date, 'month_lag_min'] = month_lag_series.min()
    store_item_sale_TS_2.loc[date, 'month_lag_variance'] = month_lag_series.var()
    
    store_item_sale_TS_2.loc[date,'week1_lag_mean'] = week1_lag_series.mean()
    store_item_sale_TS_2.loc[date,'week1_lag_max'] = week1_lag_series.max()
    store_item_sale_TS_2.loc[date,'week1_lag_min'] = week1_lag_series.min()
    store_item_sale_TS_2.loc[date,'week1_lag_variance'] = week1_lag_series.var()

