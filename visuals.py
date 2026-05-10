"""
generate_visuals.py
Author: Ahmed Ibrahim
Purpose: Generates the Exploratory Data Analysis (EDA) and Random Forest Feature 
Importance horizontal bar graphs (Figure 2 and Figure 7) used in the thesis.
"""
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("1. Loading dataset and cleaning text...")
data = pd.read_csv("Phishing_Email.csv").dropna(subset=['body', 'label'])

def clean_text(text):
    text = str(text).lower()
    return re.sub(r'[^a-z\s]', '', text)

data['cleaned_body'] = data['body'].apply(clean_text)

# Separate phishing emails for EDA
phishing_data = data[data['label'] == 1]['cleaned_body']

print("2. Performing TF-IDF Extraction...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['cleaned_body'])
y = data['label'].astype(int)

print("3. Training Random Forest to extract Feature Importance...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# --- GRAPH 1: Top Words in Phishing Emails (EDA) ---
print("4. Generating Graph 1...")
phishing_vectorizer = TfidfVectorizer(stop_words='english', max_features=15)
phishing_X = phishing_vectorizer.fit_transform(phishing_data)
phishing_scores = zip(phishing_vectorizer.get_feature_names_out(), np.asarray(phishing_X.sum(axis=0)).ravel())
sorted_phishing = sorted(phishing_scores, key=lambda x: x[1], reverse=True)
words_eda = [x[0] for x in sorted_phishing]
scores_eda = [x[1] for x in sorted_phishing]

plt.figure(figsize=(10, 6))
plt.barh(words_eda[::-1], scores_eda[::-1], color='red')
plt.title('Exploratory Data Analysis: Top 15 Most Frequent Words in Phishing Emails')
plt.xlabel('TF-IDF Score Sum')
plt.tight_layout()
plt.show()

# --- GRAPH 2: Random Forest Feature Importance ---
print("5. Generating Graph 2...")
importances = rf_model.feature_importances_
feature_names = vectorizer.get_feature_names_out()
forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)

plt.figure(figsize=(10, 6))
forest_importances.sort_values().plot(kind='barh', color='purple')
plt.title('Random Forest Feature Importance: Top 15 Words Used to Detect Phishing')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
