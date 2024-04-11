import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import sys

# UNUSED
#def plot_errors(model, X_valid, y_valid):
#    residuals = y_valid - model.predict(X_valid)
#    plt.subplot(1, 2, 1)
#    plt.hist(residuals['fire_spread_rate'])
#    plt.subplot(1, 2, 2)
#    plt.hist(residuals['duration'])
#    plt.show()
#    plt.close()

file1 = sys.argv[1]

data = pd.read_csv(file1)

# Separate data into X and y training and validation sets
X = data[['fire_year', 'current_size', 'fire_location_latitude', 'fire_location_longitude', 'fire_spread_rate', 'temperature', 'relative_humidity', 'wind_speed']].values
y = data[['size_class']].values
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# UNUSED
# convert size_class to numeric value eg. A, B, C, D, ... -> 1, 2, 3, 4, ...
# print(data['wind_direction'].unique())
# print(data['size_class'].unique())
# print(data['weather_conditions_over_fire'].unique())
# def cat2num(df):
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

    # size_class_mapping = {
        # 'A': 0, 
        # 'B': 1, 
        # 'C': 2, 
        # 'D': 3,
        # 'E': 4
    # }
    # df['size_class'] = df['size_class'].map(size_class_mapping)
    # weather_condition_mapping = {
    #     'Cloudy': 0, 
    #     'Clear': 1, 
    #     'CB Dry': 2, 
    #     'Rainshowers': 3, 
    #     'CB Wet': 4
    # }
    # df['weather_conditions_over_fire'] = df['weather_conditions_over_fire'].map(weather_condition_mapping)
    # return df

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_train = gnb.score(X_train, y_train)
gnb_valid = gnb.score(X_valid, y_valid)
# plot_errors(gnb, X_valid, y_valid)

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_train = knn.score(X_train, y_train)
knn_valid = knn.score(X_valid, y_valid)
# plot_errors(knn, X_valid, y_valid)

# Random Forest Classifier
rfc = RandomForestClassifier(max_depth=4)
rfc.fit(X_train, y_train)
rfc_train = rfc.score(X_train, y_train)
rfc_valid = rfc.score(X_valid, y_valid)

# Neural Network Classifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 6),activation='logistic', solver='lbfgs')
mlp.fit(X_train, y_train)
mlp_train = mlp.score(X_train, y_train)
mlp_valid = mlp.score(X_valid, y_valid)

# Create dataframe with ML scores
scores = pd.DataFrame({'Score': ['Training', 'Validation'], 'Naive Bayes': [gnb_train, gnb_valid], 'K Nearest Neighbours': [knn_train, knn_valid], 'Random Forest Classifier': [rfc_train, rfc_valid], 'Neural Network Classifier': [mlp_train, mlp_valid]})
scores = scores.set_index('Score')
scores.to_csv("ML_scores.csv")
