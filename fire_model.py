#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys
# Update imports as needed

# from data_clean import *

# pd.set_option('display.max_rows', None)

# Currently set to only take one input file will update as needed
file1 = 'fire_data.csv'

data = pd.read_csv(file1)

# Separate data into X and y training and validation sets
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')
data['uc_fs_date'] = pd.to_datetime(data['uc_fs_date'], errors='coerce')
data['ia_arrival_at_fire_date'] = pd.to_datetime(data['ia_arrival_at_fire_date'], errors='coerce')
mask = (-pd.isna(data['fire_start_date'])) | (-pd.isna(data['uc_fs_date']))
data = data[mask]
data['duration'] = (data['uc_fs_date'] - data['fire_start_date']).dt.total_seconds()
data['response_time'] = (data['uc_fs_date'] - data['ia_arrival_at_fire_date']).dt.total_seconds()
data = data[pd.notna(data['duration']) & pd.notna(data['response_time'])]
X = data[['fire_year', 'current_size', 'size_class', 'fire_location_latitude', 'fire_location_longitude', 'weather_conditions_over_fire', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
y = data[['fire_spread_rate', 'duration']]
# print(y)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# %%
# Linear regression
# convert size_class to numeric value eg. A, B, C, D, ... -> 1, 2, 3, 4, ...
# print(data['wind_direction'].unique())
# print(data['size_class'].unique())
# print(data['weather_conditions_over_fire'].unique())
def cat2num(df):
    # Define a dictionary mapping wind direction labels to numeric values
    # wind_direction_mapping = {
    #     'N': 0,
    #     'NE': 45,
    #     'E': 90,
    #     'SE': 135,
    #     'S': 180,
    #     'SW': 225,
    #     'W': 270,
    #     'NW': 315
    # }
    
    # # Map wind direction labels to numeric values
    # df['wind_direction'] = df['wind_direction'].map(wind_direction_mapping)
    
    # # Drop the original wind_direction column
    # df.drop(columns=['wind_direction'], inplace=True)

    size_class_mapping = {
        'A': 0, 
        'B': 1, 
        'C': 2, 
        'D': 3,
        'E': 4
    }
    df['size_class'] = df['size_class'].map(size_class_mapping)
    weather_condition_mapping = {
        'Cloudy': 0, 
        'Clear': 1, 
        'CB Dry': 2, 
        'Rainshowers': 3, 
        'CB Wet': 4
    }
    df['weather_conditions_over_fire'] = df['weather_conditions_over_fire'].map(weather_condition_mapping)
    return df

if y.isnull().any().any():
    print("Rows with NaN values:")
    print(y[y.isnull().any(axis=1)])
else:
    print('no NaN exist')

linear_model = make_pipeline(
    FunctionTransformer(cat2num), 
    LinearRegression(fit_intercept=True)
)
linear_X_train = X_train.copy()
linear_X_valid = X_valid.copy()
linear_model.fit(linear_X_train, y_train)
print(linear_model.score(linear_X_valid, y_valid))


# %%
# k nearest neighbors
# print(X_train)
# print(y_train)
kNN_model = make_pipeline(
    FunctionTransformer(cat2num),
    StandardScaler(), 
    KNeighborsRegressor(n_neighbors=5)
)
kNN_X_train = X_train.copy()
kNN_X_valid = X_valid.copy()
kNN_model.fit(kNN_X_train, y_train)
print(kNN_model.score(kNN_X_valid, y_valid))
# %%
# random forest
rf_X_train = X_train.copy()
rf_X_valid = X_valid.copy()
rf_model = make_pipeline(
    FunctionTransformer(cat2num), 
    RandomForestRegressor()    
)
# rf_model = RandomForestRegressor()
rf_model.fit(rf_X_train, y_train)
print(rf_model.score(rf_X_valid, y_valid))
# %%
# neural network regressor
mlp_X_train = X_train.copy()
mlp_X_valid = X_valid.copy()
mlp_model = make_pipeline(
    FunctionTransformer(cat2num), 
    MLPRegressor()
)
mlp_model.fit(mlp_X_train, y_train)
print(mlp_model.score(mlp_X_valid, y_valid))
# %%
