import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hmmlearn import hmm

data = pd.read_csv("fire_data.csv")

#might have to change these later
X = data[['fire_year', 'current_size', 'size_class', 'fire_location_latitude', 'fire_location_longitude', 'weather_conditions_over_fire', 'temperature', 'relative_humidity', 'wind_speed']]
y = data['fire_start_date']

#convert to binary (0: no fire, 1: fire)
is_not_na = y.notna()
y= is_not_na.astype(int)

#here i used label encoder to categorize the difference of "type" data (as opposed to difference of "kind" data)
label_encoders = {}
for column in ['size_class', 'weather_conditions_over_fire']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = hmm.GaussianHMM(n_components=8, covariance_type="diag", n_iter=1250)
model.fit(X_train)

#predict on test set
y_pred = model.predict(X_test)

#announce performance
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")