import pandas as pd

# Load CSV into a DataFrame
df = pd.read_csv("input.csv")

# Drop columns by name
columns_to_drop = ["ColumnName1", "ColumnName2"]  # replace with the actual names
df = df.drop(columns=columns_to_drop)

# Save the modified DataFrame back to CSV
df.to_csv("output.csv", index=False)
