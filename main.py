import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("1. Loading dataset and cleaning text...")
data = pd.read_csv("Phishing_Email.csv").dropna(subset=['body', 'label'])

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z\s]', '', text)

data['cleaned_body'] = data['body'].apply(clean_text)

print("2. Performing TF-IDF Extraction...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['cleaned_body'])
y = data['label'].astype(int)

print("3. Splitting into Training and Testing Sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

# --- THE COMPARATIVE STUDY ---
print("4. Training Multiple Models (Please wait ~1-2 minutes)...")

# Define the three models we are comparing
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Dictionary to store the results
results = {}

# Train and test each model
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    results[name] = {
        "Accuracy": accuracy_score(y_test, predictions) * 100,
        "Precision": precision_score(y_test, predictions) * 100,
        "Recall": recall_score(y_test, predictions) * 100,
        "F1-Score": f1_score(y_test, predictions) * 100
    }

print("\n==========================================================")
print("             COMPARATIVE STUDY RESULTS                    ")
print("==========================================================")
print(f"{'Model Name':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 58)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['Accuracy']:.2f}%     | {metrics['Precision']:.2f}%     | {metrics['Recall']:.2f}%")
print("==========================================================\n")

# --- GENERATING THE BAR GRAPH ---
print("5. Generating Bar Graph (Close the graph window to finish the script)...")

# Prepare data for the graph
names = list(results.keys())
accuracies = [results[n]['Accuracy'] for n in names]

plt.figure(figsize=(9, 5))
plt.bar(names, accuracies, color=['#4CAF50', '#2196F3', '#FFC107'])
plt.ylim(80, 100) # Zoom in on the top 20% to see differences clearly
plt.ylabel('Accuracy Percentage (%)')
plt.title('Comparison of Machine Learning Models for Phishing Detection')
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

plt.show()