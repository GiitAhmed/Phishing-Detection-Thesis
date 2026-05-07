import pandas as pd

# 1. Load the dataset
print("Loading dataset...")
data = pd.read_csv("Phishing_Email.csv")

# 2. Print the first 5 rows to make sure it works
print("Dataset loaded successfully! Here are the first 5 emails:")
print(data.head())

# 3. Print the total number of emails
print(f"\nTotal number of emails in this dataset: {len(data)}")