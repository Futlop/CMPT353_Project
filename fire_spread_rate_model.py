# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

from statsmodels.stats.multicomp import pairwise_tukeyhsd

import sys

def clean_data(file):
    # Read data and drop unwanted columns
    data = pd.read_excel(file)
    # data = data.drop(['fire_number', 'fire_name', 'fire_origin', 'industry_identifier_desc', 'responsible_group_desc', 'activity_class', 'true_cause', 'det_agent', 'det_agent_type', 'discovered_date', 'discovered_size', 'reported_date', 'dispatched_resource', 'dispatch_date', 'start_for_fire_date', 'assessment_resource', 'assessment_datetime', 'assessment_hectares', 'fire_type', 'fire_position_on_slope', 'initial_action_by', 'ia_access', 'fire_fighting_start_date', 'fire_fighting_start_size', 'bucketing_on_fire', 'distance_from_water_source', 'first_bucket_drop_date', 'bh_fs_date', 'bh_hectares', 'uc_hectares', 'to_fs_date', 'to_hectares', 'ex_fs_date', 'ex_hectares'], axis=1)

    # Filter out Null entries
    data = data[pd.notna(data['fire_year']) == True]
    data = data[pd.notna(data['current_size']) == True]
    data = data[pd.notna(data['size_class']) == True]
    data = data[pd.notna(data['fire_location_latitude']) == True]
    data = data[pd.notna(data['fire_location_longitude']) == True]
    data = data[pd.notna(data['fire_start_date']) == True]
    data = data[pd.notna(data['fire_spread_rate']) == True]
    data = data[pd.notna(data['weather_conditions_over_fire']) == True]
    data = data[pd.notna(data['temperature']) == True]
    data = data[pd.notna(data['relative_humidity']) == True]
    data = data[pd.notna(data['wind_direction']) == True]
    data = data[pd.notna(data['wind_speed']) == True]
    data = data[pd.notna(data['uc_fs_date']) == True]
    data = data[pd.notna(data['ia_arrival_at_fire_date']) == True]
    data = data[pd.notna(data['general_cause_desc']) == True]
    data = data[pd.notna(data['ex_hectares']) == True]
    return data

def plot_error(model, x, y):
    y_pred = model.predict(x)
    plt.scatter(y, y_pred)
    plt.xlabel('y_valid')
    plt.ylabel('y_pred')
    min_val = min(np.min(y), np.min(y_pred))
    max_val = max(np.max(y), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-')
    plt.show()
    plt.close()

 # %%   

# get data
file1 = 'fp-historical-wildfire-data-2006-2021.xlsx'
# data = pd.read_csv(file1)
data = clean_data(file1)
# %%
# Separate data into X and y training and validation sets
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')
data['uc_fs_date'] = pd.to_datetime(data['uc_fs_date'], errors='coerce')
data['ia_arrival_at_fire_date'] = pd.to_datetime(data['ia_arrival_at_fire_date'], errors='coerce')
mask = (-pd.isna(data['fire_start_date'])) | (-pd.isna(data['uc_fs_date']))
data = data[pd.notna(data['fire_start_date']) & pd.notna(data['uc_fs_date'])]
data['duration'] = (data['uc_fs_date'] - data['fire_start_date']).dt.total_seconds()
data['response_time'] = (data['ia_arrival_at_fire_date'] - data['fire_start_date']).dt.total_seconds()
data = data[pd.notna(data['duration']) & pd.notna(data['response_time'])]
X = data[['weather_conditions_over_fire', 'size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time', 'fire_location_latitude', 'fire_location_longitude', 'general_cause_desc',  'fire_year']]
y = data[['fire_spread_rate']]
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
# %%
# Function transformer
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

preprocessor = ColumnTransformer(
    transformers=[
        ('unorder_cat', OneHotEncoder(categories='auto', handle_unknown='ignore'), ['weather_conditions_over_fire', 'general_cause_desc']),
        ('order_cat', FunctionTransformer(cat2num), ['size_class'])
    ])
# %%
# Linear regression model
linear_model = make_pipeline(
    preprocessor, 
    LinearRegression(fit_intercept=True),
)

linear_model.fit(X_train.copy(), y_train)
print('Linear regression r^2: ', linear_model.score(X_valid.copy(), y_valid))
plot_error(linear_model, X_valid.copy(), y_valid)
# %%
# K-nearest neighbors model
kNN_model = make_pipeline(
    preprocessor,
    StandardScaler(with_mean=False), 
    KNeighborsRegressor(n_neighbors=5)
)

kNN_model.fit(X_train.copy(), y_train)
print('K-nearest neighbors r^2: ', kNN_model.score(X_valid.copy(), y_valid))
plot_error(kNN_model, X_valid.copy(), y_valid)
# %%
# Random Forest model
rf_model = make_pipeline(
    preprocessor,
    RandomForestRegressor(),
)
rf_model.fit(X_train.copy(), y_train)
model = RandomForestRegressor()
print('Random forest r^2: ', rf_model.score(X_valid.copy(), y_valid))
plot_error(rf_model, X_valid.copy(), y_valid)
# %%
# Neural Network
mlp_model = make_pipeline(
    preprocessor, 
    MLPRegressor(hidden_layer_sizes=(100, 50, 25),activation='logistic', solver='lbfgs')
)
mlp_model.fit(X_train.copy(), y_train)
print('Neural network r^2: ', mlp_model.score(X_valid.copy(), y_valid))
plot_error(mlp_model, X_valid.copy(), y_valid)
# %%
# Gradient Boosting
xgb_model = make_pipeline(
    preprocessor, 
    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
)
xgb_model.fit(X_train.copy(), y_train)
print("Gradiant boosting r^2: ", xgb_model.score(X_valid.copy(), y_valid))
plot_error(xgb_model, X_valid.copy(), y_valid)
# %%
# Feature importance
X_train_transformed = preprocessor.transform(X_train.copy())
# %%
# Feature importance: Linear regression
print('\nFeature importance: Linear regression')
linear_model_feature = LinearRegression(fit_intercept=True)
linear_model_feature.fit(X_train_transformed, y_train)
feature_names = X_train.columns
coefficients = linear_model_feature.coef_[0]

# Print feature importances (coefficients)
for feature_name, coef in zip(feature_names, coefficients):
    print(f"{feature_name}: {coef}")
# %%
# Feature importance: Random forest
print('\nFeature importance: Random forest')
rf_model_feature = RandomForestRegressor()
rf_model_feature.fit(X_train_transformed, y_train)
feature_importances_rf = rf_model_feature.feature_importances_
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importances_rf[i]}")

# %%
# Feature importance: Neural Network
print('\nFeature importance: Neural Network')
mlp_model_feature = MLPRegressor(hidden_layer_sizes=(100, 50, 25),activation='logistic', solver='lbfgs')
mlp_model_feature.fit(X_train_transformed, y_train)
coefs = mlp_model_feature.coefs_
for layer_idx, coef in enumerate(coefs):
    for neuron_idx, neuron_weights in enumerate(coef):
        for weight_idx, weight in enumerate(neuron_weights):
            print(f"Layer {layer_idx}, Neuron {neuron_idx}, Weight {weight_idx}: {weight}")      

# %%
# Feature importance: Gradiant boosting
xgb_model_feature = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
xgb_model_feature.fit(X_train_transformed, y_train)
feature_importances_xgb = xgb_model_feature.feature_importances_
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importances_xgb[i]}")
# %%
# Old model
X_itr1 = X.copy()[['size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
X_train_itr1 = X_train.copy()[['size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
X_valid_itr1 = X_valid.copy()[['size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]

# %%
# Old linear regression model
linear_model_itr1 = make_pipeline(
    FunctionTransformer(cat2num), 
    LinearRegression(fit_intercept=True),
)

linear_model_itr1.fit(X_train_itr1.copy(), y_train)
print('Old linear regression r^2: ', linear_model_itr1.score(X_valid_itr1.copy(), y_valid))
plot_error(linear_model_itr1, X_valid_itr1.copy(), y_valid)
# %%
# Old k-nearest neighbors model
kNN_model_itr1 = make_pipeline(
    FunctionTransformer(cat2num),
    StandardScaler(with_mean=False), 
    KNeighborsRegressor(n_neighbors=5)
)

kNN_model_itr1.fit(X_train_itr1.copy(), y_train)
print('Old K-nearest neighbors r^2: ', kNN_model_itr1.score(X_valid_itr1.copy(), y_valid))
plot_error(kNN_model_itr1, X_valid_itr1.copy(), y_valid)
# %%
# Old random Forest model
rf_model_itr1 = make_pipeline(
    FunctionTransformer(cat2num),
    RandomForestRegressor(),
)
rf_model_itr1.fit(X_train_itr1.copy(), y_train)
print('Old Random forest r^2: ', rf_model_itr1.score(X_valid_itr1.copy(), y_valid))
plot_error(rf_model_itr1, X_valid_itr1.copy(), y_valid)
# %%
# Old neural Network
mlp_model_itr1 = make_pipeline(
    FunctionTransformer(cat2num),
    MLPRegressor(hidden_layer_sizes=(100, 50, 25),activation='logistic', solver='lbfgs')
)
mlp_model_itr1.fit(X_train_itr1.copy(), y_train)
print('Old Neural network r^2: ', mlp_model_itr1.score(X_valid_itr1.copy(), y_valid))
plot_error(mlp_model_itr1, X_valid_itr1.copy(), y_valid)
# %%
# Old Gradient Boosting
xgb_model_itr1 = make_pipeline(
    FunctionTransformer(cat2num),
    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
)
xgb_model_itr1.fit(X_train_itr1.copy(), y_train)
print("Old Gradiant boosting r^2: ", xgb_model_itr1.score(X_valid_itr1.copy(), y_valid))
plot_error(xgb_model_itr1, X_valid_itr1.copy(), y_valid)

# %%
# Cross-validation: 
def compare_score(score1, score2):
    if stats.ttest_ind(score1, score2).pvalue > 0.05 :
        return stats.ttest_ind(score1, score2).pvalue
    else:
        return stats.mannwhitneyu(score1, score2).pvalue
def plot_hist(model_name, score1, score2):
    plt.hist(score1, bins=20, alpha=0.5, color='blue', label='New Model')
    plt.hist(score2, bins=20, alpha=0.5, color='green', label='Old Model')
    plt.xlabel('R^2 Score')
    plt.ylabel('Frequency')
    plt.title('Cross-Validation R^2 Scores: '+ str(model_name))
    plt.legend()
    plt.show()
    plt.close
# %%
# Cross-validation: Linear Regression
linear_cv_scores = cross_val_score(linear_model, X.copy(), y, cv=40, scoring='r2')
linear_itr1_cv_scores = cross_val_score(linear_model_itr1, X_itr1.copy(), y, cv=40, scoring='r2')
print('Linear regression old vs new: ', compare_score(linear_cv_scores , linear_itr1_cv_scores))
plot_hist('Linear Regression', linear_cv_scores, linear_itr1_cv_scores)

# %%
# Cross-validation: K-nearest Neighbors
knn_cv_scores = cross_val_score(kNN_model, X.copy(), y, cv=40, scoring='r2')
knn_itr1_cv_scores = cross_val_score(kNN_model_itr1, X_itr1.copy(), y, cv=40, scoring='r2')
print('K-nearest neighbors old vs new: ', compare_score(knn_cv_scores, knn_itr1_cv_scores))
plot_hist('K-Nearest Neighbors', knn_cv_scores, knn_itr1_cv_scores)

# %%
# Cross-validation: Random Forest
rf_cv_scores = cross_val_score(rf_model, X.copy(), y, cv=40, scoring='r2')
rf_itr1_cv_scores = cross_val_score(rf_model_itr1, X_itr1.copy(), y, cv=40, scoring='r2')
print('Random forest old vs new: ', compare_score(rf_cv_scores, rf_itr1_cv_scores))
plot_hist('Random Forest', rf_cv_scores, rf_itr1_cv_scores)

# %%
# Cross-validation: Neural Network
mlp_cv_scores = cross_val_score(mlp_model, X.copy(), y, cv=40, scoring='r2')
mlp_itr1_cv_scores = cross_val_score(mlp_model_itr1, X_itr1.copy(), y, cv=40, scoring='r2')
print('Neural Network old vs new: ', compare_score(mlp_cv_scores, mlp_itr1_cv_scores))
plot_hist('Neural Network', mlp_cv_scores, mlp_itr1_cv_scores)

# %%
# Cross-validation: Gradient Boosting
xgb_cv_scores = cross_val_score(xgb_model, X.copy(), y, cv=40, scoring='r2')
xgb_itr1_cv_scores = cross_val_score(xgb_model_itr1, X_itr1.copy(), y, cv=40, scoring='r2')
print('Gradient Boosting old vs new: ', compare_score(xgb_cv_scores, xgb_itr1_cv_scores))
plot_hist('Gradient Boosting', xgb_cv_scores, xgb_itr1_cv_scores)

# %%
# Cross-validation: All
anova = stats.f_oneway(linear_cv_scores, knn_cv_scores, rf_cv_scores, mlp_cv_scores)
print(anova.pvalue)
# %%
