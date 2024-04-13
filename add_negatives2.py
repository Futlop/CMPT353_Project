import pandas as pd
from datetime import timedelta

# Load the data
data = pd.read_csv('fire_data.csv')

# Convert fire_start_date to just dates (no time)
data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')

# Drop rows where dates could not be parsed
data = data.dropna(subset=['fire_start_date'])

# Floor the coordinates to group them into coordinate blocks
data['floored_latitude'] = data['fire_location_latitude'].apply(lambda x: round(x))
data['floored_longitude'] = data['fire_location_longitude'].apply(lambda x: round(x))

# Get min and max date for the date range
date_range = pd.date_range(data['fire_start_date'].min(), data['fire_start_date'].max())

# Create an empty DataFrame to store the new records
new_records = pd.DataFrame()
a = 0
# Iterate over all unique coordinate blocks
for (floored_lat, floored_lon) in data[['floored_latitude', 'floored_longitude']].drop_duplicates().itertuples(index=False):
    print(a)
    a+=1
    # Filter data for the current floored coordinate block
    block_data = data[(data['floored_latitude'] == floored_lat) & (data['floored_longitude'] == floored_lon)]

    # Get all dates that have fire data within the current floored coordinate block
    existing_dates = set(block_data['fire_start_date'])

    # Find missing dates for the current coordinate block
    missing_dates = set(date_range.date) - existing_dates

    # Add a new record for each missing date for the current coordinate block
    b=0
    for missing_date in missing_dates:
        print(b)
        b+=1
        new_record = {
            'fire_year': missing_date.year,
            'current_size': float('nan'),  # Assuming NaN for days with no fire
            'size_class': float('nan'),  # Assuming NaN for days with no fire
            'fire_location_latitude': floored_lat,
            'fire_location_longitude': floored_lon,
            'fire_start_date': missing_date,
            'fire_spread_rate': 0,  # Assuming a spread rate of 0 for days with no fire
            # Add other fields with default or NaN values as needed
        }
        new_records = new_records._append(new_record, ignore_index=True)

# Combine the new records with the original data
combined_data = pd.concat([data, new_records])

# Remove temporary columns used for flooring
combined_data = combined_data.drop(columns=['floored_latitude', 'floored_longitude'])

# Save the combined data to a CSV file
combined_data.to_csv('combined_fire_data.csv', index=False)

print("Data expansion complete.")
