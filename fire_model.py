#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import sys
# Update imports as needed

def plot_errors(model, X_valid, y_valid):

    x_list = [x for x in range(len(y_valid))]
    y_pred = model.predict(X_valid)
    # plt.scatter(x_list , y_valid.iloc[:,0])
    # plt.scatter(x_list, y_pred[:,0])
    # plt.scatter(y_valid.iloc[:,0], y_pred[:,0])
    max_y = 6e6
    plt.plot([0,max_y], [0, max_y], 'r-')
    plt.hist2d(y_valid.iloc[:,0], y_pred, 
               bins=[300, 30], cmap='Blues', norm=LogNorm())
    # plt.legend(['y_valid', 'y_pred'])
    plt.xlabel('y_valid')
    plt.ylabel('y_pred')
    plt.xlim([0, 12e6])
    plt.show()
    plt.close()

# pd.set_option('display.max_rows', None)

# Currently set to only take one input file will update as needed
file1 = 'fire_data.csv'

data = pd.read_csv(file1)

# Separate data into X and y training and validation sets
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')
data['uc_fs_date'] = pd.to_datetime(data['uc_fs_date'], errors='coerce')
data['ia_arrival_at_fire_date'] = pd.to_datetime(data['ia_arrival_at_fire_date'], errors='coerce')
mask = (-pd.isna(data['fire_start_date'])) | (-pd.isna(data['uc_fs_date']))
data = data[pd.notna(data['fire_start_date']) & pd.notna(data['uc_fs_date'])]
data['duration'] = (data['uc_fs_date'] - data['fire_start_date']).dt.total_seconds()
data['response_time'] = (data['uc_fs_date'] - data['ia_arrival_at_fire_date']).dt.total_seconds()
data = data[pd.notna(data['duration']) & pd.notna(data['response_time'])]
X = data[['size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
# y = np.log(data[['duration']] + 1e-5)
y = data[['duration']]

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

print(y_train)
# %%
# Linear regression
# convert size_class to numeric value eg. A, B, C, D, ... -> 1, 2, 3, 4, ...
# print(data['wind_direction'].unique())
# print(data['size_class'].unique())
# print(data['weather_conditions_over_fire'].unique())
def cat2num(df):
    size_class_mapping = {
        'A': 0, 
        'B': 1, 
        'C': 2, 
        'D': 3,
        'E': 4
    }
    df['size_class'] = df['size_class'].map(size_class_mapping)
    return df

def transformY(df):
    df['duration_tf'] = np.log(df['duration'] +  1e-5)
    return df

def undo_transform(x):
    np.exp(x) - 1e-5
    return x
# %%
preprocessor = ColumnTransformer(
    transformers=[
        # ('cat', OneHotEncoder(), ['weather_conditions_over_fire']),
        ('duration', FunctionTransformer(transformY), ['duration']), 
        ('size_class', FunctionTransformer(cat2num), ['size_class']),

        # Add other numerical features here if needed
        ('num', 'passthrough', ['temperature', 'relative_humidity', 'wind_speed', 'response_time'])
        # ('duration', FunctionTransformer(undo_transform), ['duration']), 
    ])

postprocessor = make_pipeline(

    FunctionTransformer(undo_transform)
)

linear_model = make_pipeline(
    FunctionTransformer(cat2num), 
    LinearRegression(fit_intercept=True),
    # FunctionTransformer(undo_transform)
)
linear_X_train = X_train.copy()
linear_X_valid = X_valid.copy()
linear_model.fit(linear_X_train, y_train)
y_pred = linear_model.predict(X_valid.copy())
FunctionTransformer(undo_transform).transform(y_pred)
print(linear_model.score(linear_X_valid, y_valid))
print(np.cor)
# plot_errors(linear_model, X_valid.copy(), y_valid)
print(y_valid)
print(y_pred)
linear_mse = mean_squared_error(y_valid, y_pred, squared=False) 
print("linear mse: ", linear_mse)

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
# print(kNN_model.score(kNN_X_valid, y_valid))
# plot_errors(kNN_model, X_valid.copy(), y_valid)
linear_mse = mean_squared_error(y_valid, kNN_model.predict(X_valid.copy()), squared=False) 
print("knn mse: ", linear_mse)
# %%
# random forest
rf_X_train = X_train.copy()
rf_y_train = y_train.copy()
rf_model = make_pipeline(
    FunctionTransformer(cat2num),
    RandomForestRegressor(),
)
rf_y_train = np.log(rf_y_train + 1e-5)
rf_model.fit(rf_X_train, rf_y_train)
y_pred = rf_model.predict(X_valid.copy())
print(y_pred)
y_pred = np.exp(y_pred) - 1e-5
print(y_pred)
print(y_valid)
# print(rf_model.score(rf_X_valid, y_valid))
# plot_errors(rf_model, X_valid.copy(), y_valid)
# rf_mse = mean_squared_error(y_valid, y_pred, squared=False) 
# print("rf mse: ", rf_mse)
r_squared = r2_score(y_valid, y_pred)
print("Coefficient of determination (R-squared):", r_squared)
# %%
# neural network regressor
mlp_X_train = X_train.copy()
mlp_X_valid = X_valid.copy()
mlp_model = make_pipeline(
    FunctionTransformer(cat2num), 

    MLPRegressor(hidden_layer_sizes=(100, 50, 25),activation='logistic', solver='lbfgs')
)
mlp_model.fit(mlp_X_train, y_train)
print(mlp_model.score(mlp_X_valid, y_valid))
# plot_errors(mlp_model, X_valid.copy(), y_valid)
# mlp_mse = mean_squared_error(y_valid, mlp_model.predict(X_valid.copy()), squared=False) 
# print("mlp mse: ", mlp_mse)
# %%
# xgboost
xgb_model = make_pipeline(
    FunctionTransformer(cat2num), 
    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
)
xgb_model.fit(X_train.copy(), y_train)
print(xgb_model.score(X_valid.copy(), y_valid))
plot_errors(xgb_model, X_valid.copy(), y_valid)
# xgb_mse = mean_squared_error(y_valid, xgb_model.predict(X_valid.copy()), squared=False) 
# print("xgb mse: ", xgb_mse)
# %%
