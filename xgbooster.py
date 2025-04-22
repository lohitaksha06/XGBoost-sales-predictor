import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Load and preprocess the data
file_path = "Hackathon-data.csv"  # Path to your CSV file
df_sales = pd.read_csv(file_path)

# Fix any column name issues
df_sales.columns = df_sales.columns.str.strip()

# Convert week to date
def convert_week_to_date(week_str):
    try:
        year = int(str(week_str)[:4])
        week = int(str(week_str)[4:])
        return pd.to_datetime(f'{year}-W{week}-1', format='%G-W%V-%u')
    except ValueError:
        print(f"Invalid week format: {week_str}")
        return pd.NaT

df_sales['date'] = df_sales['Week'].apply(convert_week_to_date)

# Feature engineering function
def add_features(df):
    df = df.sort_values(['Part ID', 'date']).copy()
    
    # Time features
    df['week_number'] = df['date'].dt.isocalendar().week
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    
    # Lag and rolling features
    for lag in [1, 2, 3]:
        df[f'lag_{lag}'] = df.groupby('Part ID')['Actual Sales'].shift(lag)
    
    for window in [3, 4]:
        df[f'rolling_mean_{window}'] = df.groupby('Part ID')['Actual Sales'].shift(1).rolling(window).mean()
    
    return df.dropna()

# Apply feature engineering
df_sales = add_features(df_sales)

# Define features and target
features = ['week_number', 'month', 'quarter', 'year', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_3', 'rolling_mean_4']
target = 'Actual Sales'

# Train and predict for each Part ID
all_predictions = []

for part_id in df_sales['Part ID'].unique():
    print(f"Processing Part ID: {part_id}")
    
    # Filter data for the current Part ID
    df_part = df_sales[df_sales['Part ID'] == part_id]
    
    # Split into features and target
    X = df_part[features]
    y = df_part[target]
    
    # Train-test split (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Train the XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"Validation MAE for Part ID {part_id}: {mae}")
    
    # Step 2: Predict sales for the next 12 weeks
    future_dates = pd.date_range(start=df_part['date'].max() + pd.Timedelta(days=7), periods=12, freq='W-MON')

    # Initialize future DataFrame
    future_data = pd.DataFrame({
        'date': future_dates,
        'week_number': [d.isocalendar()[1] for d in future_dates],  # Fix for isocalendar() compatibility
        'month': future_dates.month,
        'quarter': future_dates.quarter,
        'year': future_dates.year
    })

    # Populate lag and rolling features iteratively
    future_predictions = []
    for i in range(len(future_data)):
        # Use the last 3 actual sales and predictions as lagged values
        lag_1 = y_train.iloc[-1] if i == 0 else future_predictions[-1]
        lag_2 = y_train.iloc[-2] if i == 0 else future_predictions[-2] if i > 1 else y_train.iloc[-1]
        lag_3 = y_train.iloc[-3] if i == 0 else future_predictions[-3] if i > 2 else y_train.iloc[-2]
        
        # Calculate rolling means
        rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])
        rolling_mean_4 = np.mean([lag_1, lag_2, lag_3, y_train.iloc[-4]]) if i == 0 else np.mean(future_predictions[-4:] + [lag_1, lag_2, lag_3])

        # Create a feature row for prediction
        feature_row = pd.DataFrame({
            'week_number': [future_data.loc[i, 'week_number']],
            'month': [future_data.loc[i, 'month']],
            'quarter': [future_data.loc[i, 'quarter']],
            'year': [future_data.loc[i, 'year']],
            'lag_1': [lag_1],
            'lag_2': [lag_2],
            'lag_3': [lag_3],
            'rolling_mean_3': [rolling_mean_3],
            'rolling_mean_4': [rolling_mean_4]
        })

        # Predict the next week's sales using XGBoost
        prediction = model.predict(feature_row)[0]
        future_predictions.append(prediction)

    # Add predictions to the future DataFrame
    future_data['Predicted Sales'] = future_predictions
    future_data['Part ID'] = part_id  # Add Part ID to the predictions
    
    # Append to the list of all predictions
    all_predictions.append(future_data)

# Combine all predictions into a single DataFrame
final_predictions = pd.concat(all_predictions, ignore_index=True)

# Step 3: Save and display the results
output_file_path = "All_Parts_Future_Sales_Predictions.csv"
final_predictions.to_csv(output_file_path, index=False)

# Display the results
print("Predictions for all parts:")
print(final_predictions[['Part ID', 'date', 'Predicted Sales']])