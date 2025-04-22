import pandas as pd
import numpy as np
import time


# Specify the path to your Excel file
file_path = 'Hackathon-data.csv' 

start_time = time.time()
df = pd.read_csv(file_path)
end_time = time.time()
print(f"Time taken to load the dataset: {end_time - start_time:.2f} seconds")


# print(df.head(3))
#       Part ID    Week Region  Actual Sales 
# 0  1000A30201  202001   Asia          160.0
# 1  1000A30201  202002   Asia          110.0
# 2  1000A30201  202003   Asia          600.0

# # Convert Week to datetime
df['Date'] = pd.to_datetime(df['Week'].astype(str) + '-1', format='%Y%W-%w')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week_of_Year'] = df['Date'].dt.isocalendar().week
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
# Encode Week_of_Year
df['Sin_Week'] = np.sin(2 * np.pi * df['Week_of_Year'] / 52)
df['Cos_Week'] = np.cos(2 * np.pi * df['Week_of_Year'] / 52)
# Encode Month
df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)
# Encode Day_of_Week
df['Sales_Lag1'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(1)  # Previous week
df['Sales_Lag2'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(2)  # Two weeks ago
df['Sales_Lag3'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(3)  # Three weeks ago
df['Sales_Same_Week_Last_Year'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(52)  # Same week last year

print("=================")
print("Final DataFrame:")
print("=================")
print(df.head(3))

# Save the cleaned data to a new CSV file
cleaned_file_path = 'ramesh-predicted-Hackathon-data.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"Cleaned data saved to: {cleaned_file_path}")