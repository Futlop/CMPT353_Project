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


features_to_remove = ['wind_speed', 'temperature', 'relative_humidity','fire_spread_rate','current_size']

model_scores_by_feature_removal = {}

data = pd.read_csv('fire_data.csv')
X = data[['fire_year', 'current_size', 'fire_location_latitude', 'fire_location_longitude', 'fire_spread_rate', 'temperature', 'relative_humidity', 'wind_speed']]
y = data[['size_class']]

for feature in features_to_remove:
    X_reduced = X.drop(columns=[feature])
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_reduced, y)
    
    scores = {}
    
    for model_name, model in zip(['Naive Bayes', 'KNN', 'Random Forest', 'Neural Network'],
                                 [GaussianNB(), KNeighborsClassifier(n_neighbors=5), 
                                  RandomForestClassifier(max_depth=4), MLPClassifier(hidden_layer_sizes=(8, 6), activation='logistic', solver='lbfgs')]):
        
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        valid_score = model.score(X_valid, y_valid)
        
        scores[model_name] = {'Training': train_score, 'Validation': valid_score}
    
    model_scores_by_feature_removal[feature] = scores

print(model_scores_by_feature_removal)