import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 1: Import the dataset
file_path = 'ramesh-predicted-Hackathon-data.csv'
df = pd.read_csv(file_path)

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Apply consistent log1p transformation
df['Log_Sales'] = np.log1p(df['Actual Sales'])

# Sort the dataframe by Date (important for time series)
df = df.sort_values('Date').reset_index(drop=True)

# Step 2: Create lag and rolling features
for lag in [1, 2, 3, 4, 8]:
    df[f'Sales_Lag{lag}'] = df['Actual Sales'].shift(lag)

df['Rolling_Mean_4'] = df['Actual Sales'].shift(1).rolling(window=4).mean()
df['Rolling_Mean_8'] = df['Actual Sales'].shift(1).rolling(window=8).mean()
df['Rolling_Mean_12'] = df['Actual Sales'].shift(1).rolling(window=12).mean()
df['Rolling_Sum_4'] = df['Actual Sales'].shift(1).rolling(window=4).sum()
df['Rolling_Std_4'] = df['Actual Sales'].shift(1).rolling(window=4).std()

# Fill remaining NaNs after shifting
df = df.fillna(0)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Part ID', 'Region'], drop_first=True)

# Extract date features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week_of_Year'] = df['Date'].dt.isocalendar().week
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
df['Sin_Week'] = np.sin(2 * np.pi * df['Week_of_Year'] / 52)
df['Cos_Week'] = np.cos(2 * np.pi * df['Week_of_Year'] / 52)
df['Sin_Month'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Cos_Month'] = np.cos(2 * np.pi * df['Month'] / 12)

# Step 3: Train/test split
train_data = df[df['Date'] < '2023-01-01']
val_data = df[(df['Date'] >= '2023-01-01') & (df['Date'] < '2023-04-01')]
test_data = df[df['Date'] >= '2023-04-01']

# Define features/target
features = df.columns.difference(['Actual Sales', 'Log_Sales', 'Date'])
X_train = train_data[features]
y_train = train_data['Log_Sales']
X_val = val_data[features]
y_val = val_data['Log_Sales']

# Ensure columns match
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Step 4: Train model
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,
    'min_child_weight': 1,
    'eval_metric': 'rmse'
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=[(dval, 'validation')],
    early_stopping_rounds=50,
    verbose_eval=10
)

# Step 5: Evaluate model
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Step 6: Forecast next 12 weeks with updated rolling/lag features
def prepare_next_3_months_features(df, model, steps=12):
    future_data = df.copy()
    predictions = []

    for i in range(steps):
        last_row = future_data.iloc[-1].copy()
        next_date = last_row['Date'] + pd.Timedelta(weeks=1)

        # Build new row
        new_row = {
            'Date': next_date,
            'Year': next_date.year,
            'Month': next_date.month,
            'Week_of_Year': next_date.isocalendar().week,
            'Day_of_Week': next_date.dayofweek,
            'Quarter': next_date.quarter,
            'Sin_Week': np.sin(2 * np.pi * next_date.isocalendar().week / 52),
            'Cos_Week': np.cos(2 * np.pi * next_date.isocalendar().week / 52),
            'Sin_Month': np.sin(2 * np.pi * next_date.month / 12),
            'Cos_Month': np.cos(2 * np.pi * next_date.month / 12),
        }

        # Copy one-hot encoded columns from last row
        for col in future_data.columns:
            if col.startswith('Part ID_') or col.startswith('Region_'):
                new_row[col] = last_row[col]

        # Simulate Actual Sales using the last prediction or actual
        simulated_sales = predictions[-1] if predictions else last_row['Actual Sales']
        future_sales = future_data['Actual Sales'].tolist() + predictions

        # Add lag features
        for lag in [1, 2, 3, 4, 8]:
            new_row[f'Sales_Lag{lag}'] = future_sales[-lag] if len(future_sales) >= lag else 0

        # Add rolling features
        temp_sales = pd.Series(future_sales[-12:])  # for rolling up to 12
        new_row['Rolling_Mean_4'] = temp_sales[-4:].mean() if len(temp_sales) >= 4 else 0
        new_row['Rolling_Mean_8'] = temp_sales[-8:].mean() if len(temp_sales) >= 8 else 0
        new_row['Rolling_Mean_12'] = temp_sales.mean()
        new_row['Rolling_Sum_4'] = temp_sales[-4:].sum() if len(temp_sales) >= 4 else 0
        new_row['Rolling_Std_4'] = temp_sales[-4:].std() if len(temp_sales) >= 4 else 0

        # Fill remaining columns
        for col in features:
            if col not in new_row:
                new_row[col] = 0

        # Create DataFrame and align columns
        new_X = pd.DataFrame([new_row])[features]
        dnext = xgb.DMatrix(new_X)
        predicted_log_sales = model.predict(dnext)[0]
        predicted_sales = np.expm1(predicted_log_sales)  # inverse log1p
        predictions.append(predicted_sales)

        # Append predicted row to future_data for next iteration
        new_row['Actual Sales'] = predicted_sales
        new_row['Log_Sales'] = predicted_log_sales
        future_data = pd.concat([future_data, pd.DataFrame([new_row])], ignore_index=True)

    return predictions

# Predict future
next_3_months_sales = prepare_next_3_months_features(df, model, steps=12)
print("Predicted Sales for the Next 3 Months:", next_3_months_sales)
