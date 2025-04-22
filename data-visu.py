import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Hackathon-data.csv')

# Convert 'Week' to datetime format (you can construct the start of the week from the 'Week' column)
df['Date'] = pd.to_datetime(df['Week'].astype(str) + '0', format='%Y%U%w')  # Assuming 'Week' is in format 'YYYYWW'

# Filter for data from October 2022 onward
df_filtered = df[df['Date'] >= '2022-10-01'].copy()

# Cap (clip) individual part sales at 70,000 before aggregation
df_filtered['Actual Sales'] = df_filtered['Actual Sales'].clip(upper=70000)

# Group by month and sum sales
monthly_sales = df_filtered.groupby(df_filtered['Date'].dt.to_period('M'))['Actual Sales'].sum().reset_index()

# Sort by month
monthly_sales = monthly_sales.sort_values('Date')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales['Date'].astype(str), monthly_sales['Actual Sales'], marker='o', linestyle='-')
plt.title('Total Monthly Sales from October 2022')
plt.xlabel('Month')
plt.ylabel('Total Sales')

# Customize x-ticks to display month names
plt.xticks(rotation=45)

# Grid and layout
plt.grid(True)
plt.tight_layout()
plt.show()
