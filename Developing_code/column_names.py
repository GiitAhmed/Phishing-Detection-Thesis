import pandas as pd

# Load the dataset
print("Loading dataset...")
data = pd.read_csv("Phishing_Email.csv")

# Print the exact names of all 7 columns
print("\nHere are the column names in your dataset:")
print(data.columns.tolist())

# Print just one full row so we can see what the data looks like
print("\nHere is the data for the very first email:")
print(data.iloc[0])