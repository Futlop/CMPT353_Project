#%%
import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

import pytz
from timezonefinder import TimezoneFinder
from datetime import datetime, timedelta
#%%
# # Setup the Open-Meteo API client with cache and retry on error
# cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
# retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
# openmeteo = openmeteo_requests.Client(session = retry_session)

# latitude = 49.16
# longitude = -123.13
# start_time = "2024-03-27T01:00"
# end_time = "2024-03-27T23:00"
# # Make sure all required weather variables are listed here
# # The order of variables in hourly or daily is important to assign them correctly below
# url = "https://api.open-meteo.com/v1/forecast"
# params = {
# 	"latitude": latitude,
# 	"longitude": longitude,
# 	"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "snowfall", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", "wind_speed_180m"], 
#     "start_hour": start_time, 
#     "end_hour": end_time
# }
# responses = openmeteo.weather_api(url, params=params)

# # Process first location. Add a for-loop for multiple locations or weather models
# response = responses[0]
# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
# print(f"Elevation {response.Elevation()} m asl")
# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

# # Process hourly data. The order of variables needs to be the same as requested.
# hourly = response.Hourly()
# hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
# hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
# hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
# hourly_rain = hourly.Variables(3).ValuesAsNumpy()
# hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
# hourly_wind_speed_10m = hourly.Variables(5).ValuesAsNumpy()
# hourly_wind_speed_80m = hourly.Variables(6).ValuesAsNumpy()
# hourly_wind_speed_120m = hourly.Variables(7).ValuesAsNumpy()
# hourly_wind_speed_180m = hourly.Variables(8).ValuesAsNumpy()

# hourly_data = {"date": pd.date_range(
# 	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
# 	end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
# 	freq = pd.Timedelta(seconds = hourly.Interval()),
# 	inclusive = "left"
# )}
# hourly_data["temperature_2m"] = hourly_temperature_2m
# hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
# hourly_data["precipitation"] = hourly_precipitation
# hourly_data["rain"] = hourly_rain
# hourly_data["snowfall"] = hourly_snowfall
# hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
# hourly_data["wind_speed_80m"] = hourly_wind_speed_80m
# hourly_data["wind_speed_120m"] = hourly_wind_speed_120m
# hourly_data["wind_speed_180m"] = hourly_wind_speed_180m

# hourly_dataframe = pd.DataFrame(data = hourly_data)
# print(hourly_dataframe)
#%%
def utc2local(latitude, longitude, utc_date, utc_time):
    # Find timezone
	tf = TimezoneFinder()
	timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
      
	# change utc time format
	utc_datetime = datetime.strptime(utc_date + utc_time, '%Y-%m-%d%H%M')

    # Convert UTC time to local time
	utc_datetime = datetime.strptime(utc_datetime, '%Y-%m-%d %H:%M:%S')
	utc_datetime = pytz.utc.localize(utc_datetime)
	local_tz = pytz.timezone(timezone_str)
	local_datetime = utc_datetime.astimezone(local_tz)
	return local_datetime

#%%
def getWeatherData(lat, lng, date, time):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	latitude = lat
	longitude = lng
	start_time = datetime.strptime(date + time, '%Y-%m-%d%H%M').isoformat()
	end_time = start_time
	# Make sure all required weather variables are listed here
	# The order of variables in hourly or daily is important to assign them correctly below
	url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_80m"], 
		"start_hour": start_time, 
		"end_hour": end_time
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	
	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()[0]
	hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()[0]
	hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()[0]
	hourly_wind_speed_80m = hourly.Variables(3).ValuesAsNumpy()[0]
	return hourly_temperature_2m, hourly_relative_humidity_2m, hourly_precipitation, hourly_wind_speed_80m

print(getWeatherData(49.16, -123.13, '2024-03-29', '1900'))
# %%
def main(): 
	df = pd.read_csv('canada-2024-feb.csv')
	temperature_arr = []
	relative_humidity_arr= []
	precipitation_arr = []
	wind_speed_arr = []
	for index, r in df.iterrows():
		# Access row data using row['column_name']
		temperature, relative_humidity, precipitation, wind_speed = getWeatherData(r['latitude'], r['longitude'], r['acq_date'], str(r['acq_time']))
		temperature_arr.append(temperature)
		relative_humidity_arr.append(relative_humidity)
		precipitation_arr.append(precipitation)
		wind_speed_arr.append(wind_speed)
	# df[['temperature', 'relative_humidity', 'precipitation', 'wind_speed']]  = df.apply(lambda r: getWeatherData(r['latitude'], r['longitude'], r['acq_date'], str(r['acq_time'])), axis=1)
	df['temperature'] = temperature_arr
	df['relative_humidity'] = relative_humidity_arr
	df['precipitation'] = precipitation_arr
	df['wind_speed'] = wind_speed_arr
	print(df)	

if __name__ == '__main__':
	main()
# %%
