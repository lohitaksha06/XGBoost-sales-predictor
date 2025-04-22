import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from datetime import timedelta

# Load data
df = pd.read_csv('ramesh-predicted-Hackathon-data.csv', parse_dates=['Date'])
df['Date'] = pd.to_datetime(df['Date'])

# Cap sales at 70,000
df['Actual Sales'] = df['Actual Sales'].clip(upper=70000)

# Sum total sales per week
agg = df.groupby('Date').agg({
    'Actual Sales': 'sum',
    'Sales_Lag1': 'sum',
    'Sales_Lag2': 'sum',
    'Sales_Lag3': 'sum',
    'Sales_Lag4': 'sum',
    'Sales_Lag8': 'sum',
    'Rolling_Mean_4': 'mean',
    'Rolling_Mean_8': 'mean',
    'Rolling_Sum_4': 'sum',
    'Rolling_Std_4': 'mean',
    'Sin_Week': 'mean',
    'Cos_Week': 'mean',
    'Sin_Month': 'mean',
    'Cos_Month': 'mean',
    'Quarter': 'first',
    'Week_of_Year': 'first'
}).reset_index()

# Drop NA (initial rows without lag/rolling features)
agg = agg.dropna().reset_index(drop=True)

# Define features
features = [
    'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag4',
    'Sales_Lag8', 'Rolling_Mean_4', 'Rolling_Mean_8',
    'Rolling_Sum_4', 'Rolling_Std_4',
    'Sin_Week', 'Cos_Week', 'Sin_Month', 'Cos_Month',
    'Quarter', 'Week_of_Year'
]

# Train XGBoost
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(agg[features], agg['Actual Sales'])

# Predict next 12 weeks
predicted_sales = []
last_row = agg.iloc[-1].copy()

recent_sales = list(agg['Actual Sales'][-8:])  # use 8 because of lag8

for i in range(12):
    next_date = last_row['Date'] + timedelta(weeks=1)
    new_row = last_row.copy()
    new_row['Date'] = next_date

    # Update lag values
    new_row['Sales_Lag1'] = recent_sales[-1]
    new_row['Sales_Lag2'] = recent_sales[-2]
    new_row['Sales_Lag3'] = recent_sales[-3]
    new_row['Sales_Lag4'] = recent_sales[-4]
    new_row['Sales_Lag8'] = recent_sales[-8]

    # Rolling features from recent sales
    new_row['Rolling_Mean_4'] = np.mean(recent_sales[-4:])
    new_row['Rolling_Mean_8'] = np.mean(recent_sales[-8:])
    new_row['Rolling_Sum_4'] = np.sum(recent_sales[-4:])
    new_row['Rolling_Std_4'] = np.std(recent_sales[-4:])

    # Date-based features
    week = next_date.isocalendar().week
    new_row['Week_of_Year'] = week
    new_row['Quarter'] = pd.Timestamp(next_date).quarter
    new_row['Sin_Week'] = np.sin(2 * np.pi * week / 52)
    new_row['Cos_Week'] = np.cos(2 * np.pi * week / 52)
    new_row['Sin_Month'] = np.sin(2 * np.pi * next_date.month / 12)
    new_row['Cos_Month'] = np.cos(2 * np.pi * next_date.month / 12)

    # Predict
    pred = model.predict(pd.DataFrame([new_row])[features])[0]
    # pred = min(pred, 70000)
    new_row['Actual Sales'] = pred
    predicted_sales.append(new_row)
    
    recent_sales.append(pred)
    if len(recent_sales) > 8:
        recent_sales.pop(0)

    last_row = new_row

# Convert list of predicted sales rows into DataFrame
future_df = pd.DataFrame(predicted_sales)


# === Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(agg['Date'], agg['Actual Sales'], label='Historical Sales', color='blue', marker='o')
plt.plot(future_df['Date'], future_df['Actual Sales'], label='Predicted Sales', color='orange', linestyle='--', marker='o')

plt.title('Total Weekly Sales Forecast (Capped at 70,000)')
plt.xlabel('Week')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()
