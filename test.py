import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_emails_from_folder(folder_path, label):
    emails = []
    labels = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                emails.append(content)
                labels.append(label)
    return emails, labels

def preprocess_email(content):
    content = content.lower()
    content = re.sub(r'\d+', 'NUMBER', content)
    content = re.sub(r'http\S+', 'URL', content)
    content = re.sub(r'\W+', ' ', content)
    return content

# Load emails from folders
easy_ham_emails, easy_ham_labels = load_emails_from_folder('Data/easy_ham', 0)
hard_ham_emails, hard_ham_labels = load_emails_from_folder('Data/hard_ham', 0)
spam_emails, spam_labels = load_emails_from_folder('Data/spam', 1)

all_emails = easy_ham_emails + hard_ham_emails + spam_emails
all_labels = easy_ham_labels + hard_ham_labels + spam_labels

# Preprocess emails
processed_emails = [preprocess_email(email) for email in all_emails]

# Split data
X_train_text, X_test_text, y_train, y_test = train_test_split(processed_emails, all_labels, test_size=0.2, random_state=42)

# Set up a pipeline with CountVectorizer and Naive Bayes
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Define parameter grid for GridSearchCV
param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2)],      # Unigram, bigram, and trigram combinations
    'vectorizer__stop_words': [None, 'english'],              # With and without English stop words
    'vectorizer__max_df': [0.5, 0.7, 1.0],                   # Filter for high-frequency terms
    'vectorizer__min_df': [1, 2, 5],                         # Filter for low-frequency terms
    'vectorizer__binary': [True, False],                     # Binary and non-binary features
    'classifier__alpha': [0.001, 0.01, 0.1, 1.0]            # Different alpha values for Naive Bayes
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=3, verbose=1, n_jobs=-1)
grid_search.fit(X_train_text, y_train)

# Best parameters and model
print("Beste Parameter:", grid_search.best_params_)
print("Beste Genauigkeit:", grid_search.best_score_)

# Vectorize with the best parameters
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test_text)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Plot histogram of predicted probabilities for spam
y_prob = best_pipeline.predict_proba(X_test_text)[:, 1]
spam_threshold = 0.1
plt.figure(figsize=(10, 6))
plt.hist(y_prob, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
plt.axvline(spam_threshold, color='red', linestyle='--', label=f'Spam Threshold ({spam_threshold})')
plt.title("Histogram of Predicted Spam Probabilities")
plt.xlabel("Predicted Probability of Spam")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
