import pandas as pd

# Read CSV file
df = pd.read_csv('Nursery.csv')

# Calculate the number of rows to keep
num_rows_10 = int(len(df) * 0.1)
num_rows_25 = int(len(df) * 0.25)
num_rows_50 = int(len(df) * 0.50)

# Keep the first 10%, 25%, 50% of the rows
df_top_10_percent = df.head(num_rows_10)
df_top_25_percent = df.head(num_rows_25)
df_top_50_percent = df.head(num_rows_50)

# Save the results to the new CSV files
df_top_10_percent.to_csv('Nursery_10%.csv', index=False)
df_top_25_percent.to_csv('Nursery_25%.csv', index=False)
df_top_50_percent.to_csv('Nursery_50%.csv', index=False)
