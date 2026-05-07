import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("1. Loading dataset...")
data = pd.read_csv("Phishing_Email.csv")

# Sometimes datasets have blank rows. Let's drop any rows missing a body or label.
data = data.dropna(subset=['body', 'label'])

print("2. Cleaning the text data (Section 3.3)...")
def clean_text(text):
    text = str(text).lower() # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters, HTML, and numbers
    return text

# Apply the cleaning function to every email in the dataset
data['cleaned_body'] = data['body'].apply(clean_text)

print("3. Performing TF-IDF Feature Extraction (Section 3.4)...")
# TfidfVectorizer automatically removes 'stop words' (the, is, at) for us!
# We limit it to the top 5000 most important words to keep the model 'lightweight' for SMEs.
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['cleaned_body'])
y = data['label'].astype(int) # Ensure labels are treated as integers (0 or 1)

print("4. Splitting data into 80% Training and 20% Testing (Section 3.5)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("5. Training the Random Forest Model (This might take 1 to 3 minutes!)...")
# n_jobs=-1 tells the computer to use all its core processors to train faster
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("6. Testing the model and calculating metrics (Section 3.6)...")
predictions = rf_model.predict(X_test)

print("\n==================================")
print("          FINAL RESULTS           ")
print("==================================")
print(f"Accuracy:  {accuracy_score(y_test, predictions) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, predictions) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, predictions) * 100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, predictions) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("==================================")