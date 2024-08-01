import pandas as pd

# Read CSV file
df = pd.read_csv('nursery.csv')

# Calculate the number of rows to keep
num_rows = int(len(df) * 0.3)

# Keep the first 30% of the rows
df_top_30_percent = df.head(num_rows)

# Keep the first 4 columns and save as a new CSV file
df_top_30_percent_4_cols = df_top_30_percent.iloc[:, :4]
df_top_30_percent_4_cols.to_csv('Nursery_4attributes.csv', index=False)

# Keep the first 6 columns and save as a new CSV file
df_top_30_percent_6_cols = df_top_30_percent.iloc[:, :6]
df_top_30_percent_6_cols.to_csv('Nursery_6attributes.csv', index=False)

# Keep the first 8 columns and save as a new CSV file
df_top_30_percent_8_cols = df_top_30_percent.iloc[:, :8]
df_top_30_percent_8_cols.to_csv('Nursery_8attributes.csv', index=False)
