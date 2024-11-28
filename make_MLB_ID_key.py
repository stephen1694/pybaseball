# Import modules
import pandas as pd
import glob

# Set up:
# downoladed people files from https://github.com/chadwickbureau/register/tree/master/data
# to chadwick_register/ in my working directory

# Define directory and file pattern to search & import
file_pattern = "chadwick_register/people-?.csv"

# Use glob to find all matching files
file_list = glob.glob(file_pattern)

# Read and concatenate the CSV files specifiying that cols 8-10 are strings
df = pd.concat(
    (pd.read_csv(file, dtype={8: str, 9: str, 10: str}) for file in file_list),
    ignore_index=True
)

# Filter for rows where 'key_mlbam' is not null (retain only MLB related people) and 'birth_year' > 1900
filtered_df = df[df['key_mlbam'].notna() & (df['birth_year'] > 1900)]

# drop manager & upire info columns
filtered_df = filtered_df.drop(filtered_df.columns[30:40], axis=1)

# Delete original DataFrame
del df

# Inspect resulting DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(filtered_df.info())
print(filtered_df.head())

# Write out filtered_df as MLB_ID_key.csv
filtered_df.to_csv('MLB_ID_key.csv')
