# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
def cat2num(df):
    wind_direction_mapping = {
        'N': 0,
        'NE': 45,
        'E': 90,
        'SE': 135,
        'S': 180,
        'SW': 225,
        'W': 270,
        'NW': 315
    }
    df['wind_direction'] = df['wind_direction'].map(wind_direction_mapping)
    df.drop(columns=['wind_direction'], inplace=True)
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


# Update imports as needed

# Currently set to only take one input file will update as needed
file1 = 'fire_data.csv'

data = pd.read_csv(file1)

# TODO: Transform data and run statistical tests
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')
data['uc_fs_date'] = pd.to_datetime(data['uc_fs_date'], errors='coerce')
data['ia_arrival_at_fire_date'] = pd.to_datetime(data['ia_arrival_at_fire_date'], errors='coerce')
mask = (-pd.isna(data['fire_start_date'])) | (-pd.isna(data['uc_fs_date']))
data = data[pd.notna(data['fire_start_date']) & pd.notna(data['uc_fs_date'])]
data['duration'] = (data['uc_fs_date'] - data['fire_start_date']).dt.total_seconds()
data['response_time'] = (data['uc_fs_date'] - data['ia_arrival_at_fire_date']).dt.total_seconds()
data = data[pd.notna(data['duration']) & pd.notna(data['response_time'])]
FunctionTransformer(cat2num).transform(data)
X = data[['current_size', 'size_class', 'weather_conditions_over_fire', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
y = data[['fire_spread_rate', 'duration']]

# Feature analysis: Correlation test
# current_size vs fire_spread_rate
correlation_current_size_fire_spread_rate = data['current_size'].corr(data['fire_spread_rate'])
print("Correlation between current_size and fire_spread_rate:", correlation_current_size_fire_spread_rate)

# current_size vs duration
correlation_current_size_duration = data['current_size'].corr(data['duration'])
print("Correlation between current_size and duration:", correlation_current_size_duration)

# size_class vs fire_spread_rate
correlation_size_class_fire_spread_rate = data['size_class'].corr(data['fire_spread_rate'])
print("Correlation between size_class and fire_spread_rate:", correlation_size_class_fire_spread_rate)

# size_class vs duration
correlation_size_class_duration = data['size_class'].corr(data['duration'])
print("Correlation between size_class and duration:", correlation_size_class_duration)

# weather_conditions_over_fire vs fire_spread_rate
correlation_weather_conditions_fire_spread_rate = data['weather_conditions_over_fire'].corr(data['fire_spread_rate'])
print("Correlation between weather_conditions_over_fire and fire_spread_rate:", correlation_weather_conditions_fire_spread_rate)

# weather_conditions_over_fire vs duration
correlation_weather_conditions_duration = data['weather_conditions_over_fire'].corr(data['duration'])
print("Correlation between weather_conditions_over_fire and duration:", correlation_weather_conditions_duration)

# temperature vs fire_spread_rate
correlation_temperature_fire_spread_rate = data['temperature'].corr(data['fire_spread_rate'])
print("Correlation between temperature and fire_spread_rate:", correlation_temperature_fire_spread_rate)

# temperature vs duration
correlation_temperature_duration = data['temperature'].corr(data['duration'])
print("Correlation between temperature and duration:", correlation_temperature_duration)

# relative_humidity vs fire_spread_rate
correlation_relative_humidity_fire_spread_rate = data['relative_humidity'].corr(data['fire_spread_rate'])
print("Correlation between relative_humidity and fire_spread_rate:", correlation_relative_humidity_fire_spread_rate)

# relative_humidity vs duration
correlation_relative_humidity_duration = data['relative_humidity'].corr(data['duration'])
print("Correlation between relative_humidity and duration:", correlation_relative_humidity_duration)

# wind_speed vs fire_spread_rate
correlation_wind_speed_fire_spread_rate = data['wind_speed'].corr(data['fire_spread_rate'])
print("Correlation between wind_speed and fire_spread_rate:", correlation_wind_speed_fire_spread_rate)

# wind_speed vs duration
correlation_wind_speed_duration = data['wind_speed'].corr(data['duration'])
print("Correlation between wind_speed and duration:", correlation_wind_speed_duration)

# response_time vs fire_spread_rate
correlation_response_time_fire_spread_rate = data['response_time'].corr(data['fire_spread_rate'])
print("Correlation between response_time and fire_spread_rate:", correlation_response_time_fire_spread_rate)

# response_time vs duration
correlation_response_time_duration = data['response_time'].corr(data['duration'])
print("Correlation between response_time and duration:", correlation_response_time_duration)
# %%
# Feature analysis: Feature scaling
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# Extract feature importances
feature_importances_rf = rf_model.feature_importances_

# Print feature importances
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {feature_importances_rf[i]}")
# %%
def plot_errors(model, X_valid, y_valid):
    residuals = y_valid - model.predict(X_valid)
    plt.subplot(1, 2, 1)
    plt.hist(residuals['fire_spread_rate'])
    plt.subplot(1, 2, 2)
    plt.hist(residuals['duration'])
    plt.close()
# %%

import seaborn as sns

correlations = {
    'current_size': {'fire_spread_rate': 0.07904327962635205, 'duration': 0.03018501036834419},
    'size_class': {'fire_spread_rate': 0.4425313711486078, 'duration': 0.0542515909201817},
    'weather_conditions_over_fire': {'fire_spread_rate': 0.02544602061949344, 'duration': -0.017183250296072032},
    'temperature': {'fire_spread_rate': 0.14703346007542073, 'duration': -0.01341779673022259},
    'relative_humidity': {'fire_spread_rate': -0.13496297443756816, 'duration': -0.02125557984998247},
    'wind_speed': {'fire_spread_rate': 0.15571609723825824, 'duration': 0.02061171268020623},
    'response_time': {'fire_spread_rate': 0.18809034085791815, 'duration': 0.08352398933648272}
}
feature_importances = {
    'current_size': 0.14600742581278667,
    'size_class': 0.007771569288176591,
    'weather_conditions_over_fire': 0.026574156836813752,
    'temperature': 0.11155985814792427,
    'relative_humidity': 0.0640174445822936,
    'wind_speed': 0.22189227384294344,
    'response_time': 0.42217727148906165
}

corr_df = pd.DataFrame(correlations).T

plt.figure(figsize=(10, 6))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation between Features and Targets')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), list(feature_importances.values()), align='center')
plt.xticks(range(len(feature_importances)), list(feature_importances.keys()), rotation=45)
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
