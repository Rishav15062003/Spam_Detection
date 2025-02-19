import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os
from collections import Counter

# Load dataset
file_path = "/content/spam.csv"   # update the file path
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path, encoding="latin-1")

# Ensure required column exists
if "Prediction" not in df.columns:
    raise ValueError("Missing required column: Prediction")

# Extract features and labels
X = df.drop(columns=["Email No.", "Prediction"], errors='ignore')
y = df["Prediction"].astype(int)  # Ensure integer format

# Convert numerical feature names to string (in case of column mismatch)
X.columns = X.columns.astype(str)

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to convert raw email text into word count vector
def preprocess_email(email_text, feature_columns):
    if not isinstance(email_text, str) or len(email_text.strip()) == 0:
        raise ValueError("Invalid email text. It must be a non-empty string.")
    
    # Tokenize email text and count word occurrences
    words = email_text.lower().split()
    word_counts = Counter(words)
    
    # Create feature vector matching X.columns
    email_vector = [word_counts.get(word, 0) for word in feature_columns]
    return pd.DataFrame([email_vector], columns=feature_columns)

# Example usage with raw email text
custom_email = "Meeting Scheduled for Machine Learning Project on 11/02/23"
custom_email_vector = preprocess_email(custom_email, X.columns)
prediction = model.predict(custom_email_vector)
print("Custom Email Prediction:", "Spam" if prediction[0] == 1 else "Ham")
