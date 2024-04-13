#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Selecting a subset of relevant features for the model
features = ['relative_humidity', 'temperature', 'wind_speed', 'fire_year', 'current_size', 'fire_location_latitude', 'fire_location_longitude', 'general_cause_desc', 'size_class']
target = 'fire_spread_rate'

df_wildfire = pd.read_excel('fp-historical-wildfire-data-2006-2021.xlsx')

# Filtering out negative spread rates
df_model = df_wildfire[df_wildfire[target] >= 0]

# Separating features and target variable
X = df_model[features]
y = df_model[target].values

# Handling categorical variables via OneHotEncoding and missing values
categorical_features = ['general_cause_desc', 'size_class']
numerical_features = ['relative_humidity', 'temperature', 'wind_speed', 'fire_year', 'current_size', 'fire_location_latitude', 'fire_location_longitude']

numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Splitting data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# %%
print(my_pipeline.score(X_valid, y_valid))
# %%
from xgboost import XGBRegressor

# Define the XGBoost model
xgb_model = XGBRegressor()

# Integrate XGBoost into the pipeline
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', xgb_model)
                              ])

# Training the XGBoost model
xgb_pipeline.fit(X_train, y_train)

print(xgb_pipeline.score(X_valid, y_valid))
# %%
