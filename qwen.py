import pandas as pd
import numpy as np
import time


# Specify the path to your Excel file
file_path = 'Hackathon-data.csv' 

start_time = time.time()
df = pd.read_csv(file_path)
end_time = time.time()
print(f"Time taken to load the dataset: {end_time - start_time:.2f} seconds")

print(df.head(3))


# # Convert Week to datetime
df['Date'] = pd.to_datetime(df['Week'].astype(str) + '-1', format='%Y%W-%w')
# df['Log_Sales'] = np.log1p(df['Actual Sales']) 