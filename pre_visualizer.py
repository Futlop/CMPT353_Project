import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import sys
#name = sys.argv[1]
name = 'fire_data.csv'
data = pd.read_csv(name)
data['fire_spread_rate'] += 0.001#so the log operation can function


data['log_fire_spread_rate'] = np.log(data['fire_spread_rate'])

numerical_features = ['current_size', 'fire_spread_rate', 'temperature', 'relative_humidity', 'wind_speed','log_fire_spread_rate']
for feature in numerical_features:
    plt.figure(figsize=(10, 4))
    plt.hist(data[feature], bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['temperature'], data['fire_spread_rate'])
plt.title('Temperature vs. Fire Spread Rate')
plt.xlabel('Temperature')
plt.ylabel('Fire Spread Rate')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='size_class', y='current_size', data=data)
plt.title('Boxplot of Current Size by Size Class')
plt.show()

data['fire_start_date'] = pd.to_datetime(data['fire_start_date'])
plt.figure(figsize=(15, 7))
plt.plot(data['fire_start_date'], data['fire_spread_rate'])
plt.title('Time Series of Fire Spread Rate')
plt.xlabel('Date')
plt.ylabel('Fire Spread Rate')
plt.show()

weather_counts = data['weather_conditions_over_fire'].value_counts()
plt.figure(figsize=(10, 6))
weather_counts.plot(kind='bar')
plt.title('Frequency of Different Weather Conditions Over Fire')
plt.xlabel('Weather Conditions')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(data['fire_location_longitude'], data['fire_location_latitude'], c=data['current_size'])
plt.colorbar()
plt.title('Map of Fire Locations by Size')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
