def drop_outliers(data, column_name):
    q1 = data[column_name].quantile(0.25)
    q3 = data[column_name].quantile(0.75)
    IQR = q3 - q1
    lower_bound = q3 - 1.5 * IQR
    upper_bound = q3 + 1.5 * IQR
    data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    print(lower_bound, upper_bound)
    return data
