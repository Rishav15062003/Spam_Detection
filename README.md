# Spam_Detection
Summary of the Email Spam Detection Code
This Python script builds a Naïve Bayes-based email spam detector using a bag-of-words approach.

🔹 Key Steps:
Load and Prepare Data

Reads spam.csv and ensures the "Prediction" column exists.
Splits features (X) and target labels (y), removing unnecessary columns.
Train a Machine Learning Model

Uses Multinomial Naïve Bayes to classify emails as spam or ham.
Splits data into 80% training, 20% testing and trains the model.
Evaluate Model Performance

Computes accuracy and classification report to assess predictions.
Custom Email Classification

Converts new email text into a word count vector matching the dataset format.
Predicts if the email is Spam (1) or Ham (0).
