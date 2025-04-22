import pandas as pd

# Specify the path to your Excel file
excel_file_path = 'Hackathon-data.xlsx'  # Replace with the actual file path

# Load the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)

# Specify the path for the new CSV file
csv_file_path = 'Hackathon-data.csv'  # Replace with your desired output file name

# Save the DataFrame as a CSV file
df.to_csv(csv_file_path, index=False)  # `index=False` avoids writing row indices to the CSV

print(f"File successfully converted and saved as: {csv_file_path}")