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

# Define a list of lag values
lags = [1, 2, 3, 4, 8]
for lag in lags:
    df[f'Sales_Lag{lag}'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(lag)

df['Sales_Same_Week_Last_Year'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].shift(52)  # Same week last year

# Define a list of rolling window sizes
rolling_windows = [4, 8, 12] 
for window in rolling_windows:
    df[f'Rolling_Mean_{window}'] = (df.groupby(['Part ID', 'Region'])['Actual Sales'].rolling(window=window).mean().reset_index(level=[0, 1], drop=True))


df['Rolling_Sum_4'] = df.groupby(['Part ID', 'Region'])['Actual Sales'].rolling(window=4).sum().reset_index(level=[0, 1], drop=True)
df['Rolling_Std_4'] = (df.groupby(['Part ID', 'Region'])['Actual Sales'].rolling(window=4).std().reset_index(level=[0, 1], drop=True))
# Example: Count of unique Part IDs per Region
part_id_counts = df.groupby('Region')['Part ID'].nunique().reset_index(name='Part_ID_Count')
df = df.merge(part_id_counts, on='Region', how='left')
df['Log_Sales'] = np.log1p(df['Actual Sales'])  # Add 1 to handle zeros
                                    

# Save the cleaned data to a new CSV file
cleaned_file_path = 'ramesh-predicted-Hackathon-data.csv'
df.to_csv(cleaned_file_path, index=False)

# Fill missing lag values
for col in ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag4', 'Sales_Lag8','Rolling_Mean_4', 'Rolling_Mean_8', 'Rolling_Mean_12', 'Rolling_Sum_4', 'Rolling_Std_4',]:
    df[col] = df[col].fillna(0)

print("=================")
print(" Final DataFrame:")
print("=================")
print(df.head(5))


print(f"Cleaned data saved to: {cleaned_file_path}")