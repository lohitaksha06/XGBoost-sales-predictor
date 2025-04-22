import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Step 1: Import the dataset
file_path = 'ramesh-predicted-Hackathon-data.csv'
df = pd.read_csv(file_path)

# Ensure the 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Step 2: Handle missing values in lag and rolling statistics
for col in ['Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3', 'Sales_Lag4', 'Sales_Lag8']:
    df[col] = df[col].fillna(0)

for col in ['Rolling_Mean_4', 'Rolling_Mean_8', 'Rolling_Mean_12', 'Rolling_Sum_4', 'Rolling_Std_4']:
    df[col] = df[col].fillna(0)

# Step 3: Encode categorical variables ('Part ID' and 'Region')
df = pd.get_dummies(df, columns=['Part ID', 'Region'], drop_first=True)

# Step 4: Split the data into training, validation, and test sets
train_data = df[df['Date'] < '2023-01-01']  # First ~3.5 years
val_data = df[(df['Date'] >= '2023-01-01') & (df['Date'] < '2023-04-01')]  # Next 3 months
test_data = df[df['Date'] >= '2023-04-01']  # Last 3 months

# Step 5: Define features and target
X_train = train_data.drop(columns=['Actual Sales', 'Date'])
y_train = train_data['Log_Sales']  # Use log-transformed sales for training

X_val = val_data.drop(columns=['Actual Sales', 'Date'])
y_val = val_data['Log_Sales']

# Ensure X_train and X_val have the same columns
common_columns = X_train.columns.intersection(X_val.columns)
X_train = X_train[common_columns]
X_val = X_val[common_columns]

# Handle missing values
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

# Step 6: Train the XGBoost model
params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.05,
    'max_depth': 8,
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

# Step 7: Evaluate the model on the validation set
y_pred = model.predict(dval)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse}")

# Step 8: Predict the next 3 months (iteratively handle lag features)
def prepare_next_3_months_features(df, model, weeks_to_predict=12):
    """
    Iteratively predict the next 3 months of sales.
    """
    last_date = df['Date'].max()
    predictions = []

    for week in range(weeks_to_predict):
        # Create a copy of the last available data point
        next_week_data = df.iloc[[-1]].copy()

        # Update the date to the next week
        next_week_data['Date'] = last_date + pd.Timedelta(weeks=week + 1)
        next_week_data['Year'] = next_week_data['Date'].dt.year
        next_week_data['Month'] = next_week_data['Date'].dt.month
        next_week_data['Week_of_Year'] = next_week_data['Date'].dt.isocalendar().week
        next_week_data['Day_of_Week'] = next_week_data['Date'].dt.dayofweek
        next_week_data['Quarter'] = next_week_data['Date'].dt.quarter

        # Update cyclical encoding
        next_week_data['Sin_Week'] = np.sin(2 * np.pi * next_week_data['Week_of_Year'] / 52)
        next_week_data['Cos_Week'] = np.cos(2 * np.pi * next_week_data['Week_of_Year'] / 52)
        next_week_data['Sin_Month'] = np.sin(2 * np.pi * next_week_data['Month'] / 12)
        next_week_data['Cos_Month'] = np.cos(2 * np.pi * next_week_data['Month'] / 12)

        # Update lag features using the last prediction
        if week > 0:
            next_week_data['Sales_Lag1'] = predictions[-1]
            next_week_data['Sales_Lag2'] = predictions[-2] if len(predictions) > 1 else df['Actual Sales'].iloc[-1]
            next_week_data['Sales_Lag3'] = predictions[-3] if len(predictions) > 2 else df['Actual Sales'].iloc[-2]
            next_week_data['Sales_Lag4'] = predictions[-4] if len(predictions) > 3 else df['Actual Sales'].iloc[-3]
            next_week_data['Sales_Lag8'] = predictions[-8] if len(predictions) > 7 else df['Actual Sales'].iloc[-8]

        # Drop unnecessary columns
        next_week_data = next_week_data.drop(columns=['Actual Sales', 'Date'])

        # One-hot encode categorical variables again (if needed)
        next_week_data = pd.get_dummies(next_week_data, columns=[], drop_first=True)

        # Align columns with training data
        next_week_data = next_week_data.reindex(columns=common_columns, fill_value=0)

        # Predict the next week's sales
        predicted_log_sales = model.predict(xgb.DMatrix(next_week_data))
        predicted_sales = np.expm1(predicted_log_sales)  # Reverse log transformation
        predictions.append(predicted_sales[0])

    return predictions

# Predict the next 3 months (12 weeks)
next_3_months_sales = prepare_next_3_months_features(df, model, weeks_to_predict=12)
print("Predicted Sales for the Next 3 Months:", next_3_months_sales)