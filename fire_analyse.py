# %%
import numpy as np
import pandas as pd
from fire_model import *
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import matplotlib.pyplot as plt

import sys
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
X = data[['size_class', 'temperature', 'relative_humidity', 'wind_speed', 'response_time']]
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

# # weather_conditions_over_fire vs fire_spread_rate
# correlation_weather_conditions_fire_spread_rate = data['weather_conditions_over_fire'].corr(data['fire_spread_rate'])
# print("Correlation between weather_conditions_over_fire and fire_spread_rate:", correlation_weather_conditions_fire_spread_rate)

# # weather_conditions_over_fire vs duration
# correlation_weather_conditions_duration = data['weather_conditions_over_fire'].corr(data['duration'])
# print("Correlation between weather_conditions_over_fire and duration:", correlation_weather_conditions_duration)

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
# Normality test
plt.figure(figsize=(10, 20))

# Plot histograms for each feature
plt.subplot(7, 1, 1)
print('current size p-value: ', stats.normaltest(data['current_size']).pvalue)
plt.hist(data['current_size'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Current Size')

plt.subplot(7, 1, 2)
# print('size_class p-value: ', stats.normaltest(data['size_class']).pvalue)
plt.hist(data['size_class'], bins=10, color='salmon', edgecolor='black')
plt.title('Distribution of Size Class')

plt.subplot(7, 1, 3)
# print('weather_conditions_over_fire p-value: ', stats.normaltest(data['weather_conditions_over_fire']).pvalue)
plt.hist(data['weather_conditions_over_fire'], bins=5, color='green', edgecolor='black')
plt.title('Distribution of Weather Conditions Over Fire')

plt.subplot(7, 1, 4)
print('temperaturee p-value: ', stats.normaltest(data['temperature']).pvalue)
plt.hist(data['temperature'], bins=20, color='purple', edgecolor='black')
plt.title('Distribution of Temperature')

plt.subplot(7, 1, 5)
print('relative_humidity p-value: ', stats.normaltest(np.sqrt(data['relative_humidity'])).pvalue)
plt.hist(data['relative_humidity'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Relative Humidity')

plt.subplot(7, 1, 6)
print('wind_speed p-value: ', stats.normaltest(np.sqrt(data['wind_speed'])).pvalue)
plt.hist(data['wind_speed'], bins=20, color='pink', edgecolor='black')
plt.title('Distribution of Wind Speed')

plt.subplot(7, 1, 7)
print('response_time p-value: ', stats.normaltest(data['response_time']).pvalue)
plt.hist(data['response_time'], bins=20, color='gray', edgecolor='black')
plt.title('Distribution of Response Time')

plt.tight_layout()
plt.show()

# %%
