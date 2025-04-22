import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Import the dataset
file_path = 'ramesh-predicted-Hackathon-data.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply consistent log1p transformation
df['Log_Sales'] = np.log1p(df['Actual Sales'])

# Sort the dataframe by Date (important for time series)
df = df.sort_values('Date').reset_index(drop=True)

features = [
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag4',
    'Sales_Lag8', 'Rolling_Mean_4', 'Rolling_Mean_8',
    'Rolling_Sum_4', 'Rolling_Std_4', 'Part_ID_Count',
    'Sin_Week', 'Cos_Week', 'Sin_Month', 'Cos_Month',
    'Quarter', 'Week_of_Year'
]
target = 'Actual Sales'

train = df.iloc[:-12]
test = df.iloc[-12:]

model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(train[features], train[target])

future_preds = []

current_df = test.copy()
for i in range(12):
    pred = model.predict(current_df[features].iloc[[i]])[0]
    future_preds.append(pred)
    
    # Update next row lag features with this prediction
    if i + 1 < 12:
        current_df.at[current_df.index[i+1], 'Sales_Lag1'] = pred
        # Shift lag features down manually
        for lag in [2, 3, 4]:
            current_df.at[current_df.index[i+1], f'Sales_Lag{lag}'] = current_df.at[current_df.index[i], f'Sales_Lag{lag-1}']

import matplotlib.pyplot as plt
from datetime import timedelta

# Get the last known date from the dataset
last_date = df['Date'].max()

# Generate next 12 week dates
future_dates = [last_date + timedelta(weeks=i+1) for i in range(12)]

# Plot only predicted values
plt.figure(figsize=(10, 5))
plt.plot(future_dates, future_preds, label='Predicted Sales', marker='o', linestyle='--')
plt.title('Forecasted Sales for Next 12 Weeks')
plt.xlabel('Week')
plt.ylabel('Predicted Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
