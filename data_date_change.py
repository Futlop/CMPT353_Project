import pandas as pd
#This file is meant to make the job of the HMM easier. Because it can't predict the exact time with any accuracy because it is a discrete model, I have to make it so that the data
#is in a discrete format. To do this, I will 
data = pd.read_csv('fire_data.csv')

data['fire_start_date'] = pd.to_datetime(data['fire_start_date'], errors='coerce')

#convert to weeks
data_week = data.copy()
data_week['fire_start_date'] = data_week['fire_start_date'].dt.isocalendar().week

#convert to months
data_month = data.copy()
data_month['fire_start_date'] = data_month['fire_start_date'].dt.month

data_week.to_csv('fire_data_week.csv', index=False)
data_month.to_csv('fire_data_month.csv', index=False)

print("saved as 'fire_data_week.csv'")
print("saved as 'fire_data_month.csv'")
