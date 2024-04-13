#%%
import numpy as np
import pandas as pd
import sys

file = 'fp-historical-wildfire-data-2006-2021.xlsx'

# Read data and drop unwanted columns
data = pd.read_excel(file)
data = data.drop(['fire_number', 'fire_name', 'fire_origin', 'general_cause_desc', 'industry_identifier_desc', 'responsible_group_desc', 'activity_class', 'true_cause', 'det_agent', 'det_agent_type', 'discovered_date', 'discovered_size', 'reported_date', 'dispatched_resource', 'dispatch_date', 'start_for_fire_date', 'assessment_resource', 'assessment_datetime', 'assessment_hectares', 'fire_type', 'fire_position_on_slope', 'fuel_type', 'initial_action_by', 'ia_access', 'fire_fighting_start_date', 'fire_fighting_start_size', 'bucketing_on_fire', 'distance_from_water_source', 'first_bucket_drop_date', 'bh_fs_date', 'bh_hectares', 'uc_hectares', 'to_fs_date', 'to_hectares', 'ex_fs_date', 'ex_hectares'], axis=1)

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

# for finding response time
data = data[pd.notna(data['ia_arrival_at_fire_date']) == True]

# Write to csv
data.to_csv("fire_data.csv")

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
    data = data[pd.notna(data['fuel_type']) == True]
    return data

# %%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])