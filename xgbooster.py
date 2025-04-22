import pandas as pd

# Load the CSV file
file_path = "Hackathon-data.csv"
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

# Save the updated DataFrame to a new CSV file
output_file_path = "Lohit-predicted_Sales_Data.csv"
df_sales.to_csv(output_file_path, index=False)

print(f"Updated data saved to {output_file_path}")