import pandas as pd
from datetime import datetime, timedelta

data = pd.read_csv('fire_data.csv')
#convert the fire_start_date to datetime, removing the time component
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'],errors='coerce')

#get range of dates
start_date = data['fire_start_date'].min()
end_date = data['fire_start_date'].max()

#generate a list of all dates within the range
all_dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

#get a list of all unique coordinate pairs
unique_coords = data[['fire_location_latitude', 'fire_location_longitude']].drop_duplicates()

#make a dataframe to hold the expanded data
expanded_data = pd.DataFrame()

#iterate through all dates and coordinates
for date in all_dates:
    for _, coords in unique_coords.iterrows():
        if not data[(data['fire_start_date'] == date)
                    & (data['fire_location_latitude'] == coords['fire_location_latitude'])
                    & (data['fire_location_longitude'] == coords['fire_location_longitude'])].empty:
            #fire data for this date and coordinate pair exists, so skip
            continue
        #no fire data for this date and coordinate pair, so create a new record
        new_record = {'fire_year': date.year, 'fire_start_date': date,
                      'fire_location_latitude': coords['fire_location_latitude'],
                      'fire_location_longitude': coords['fire_location_longitude'],
                      #add other necessary columns with default values here
                      'fire_spread_rate': 0,  # or np.nan as appropriate
                      
                      }
        #append the new record to the expanded_data df
        expanded_data = expanded_data._append(new_record, ignore_index=True)

#combine the original data with the expanded data
combined_data = pd.concat([data, expanded_data], ignore_index=True)

combined_data.to_csv('expanded_fire_data.csv', index=False)

