import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load emails from a specified folder and label them
def load_emails_from_folder(folder_path, label):
    emails = []
    labels = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Load each email
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                emails.append(content)
                labels.append(label)
    return emails, labels

# Paths to each folder
easy_ham_path = 'Data/easy_ham'
hard_ham_path = 'Data/hard_ham'
spam_path = 'Data/spam'

# Load emails from each folder
easy_ham_emails, easy_ham_labels = load_emails_from_folder(easy_ham_path, 0)
hard_ham_emails, hard_ham_labels = load_emails_from_folder(hard_ham_path, 0)
spam_emails, spam_labels = load_emails_from_folder(spam_path, 1)

# Combine all emails and labels
all_emails = easy_ham_emails + hard_ham_emails + spam_emails
all_labels = easy_ham_labels + hard_ham_labels + spam_labels

###########################################################################################################################################

# Function to preprocess email content
def preprocess_email(content):
    content = content.lower()
    content = re.sub(r'\d+', 'NUMBER', content)
    content = re.sub(r'http\S+', 'URL', content)
    content = re.sub(r'\W+', ' ', content)
    return content

# Preprocess all emails
processed_emails = [preprocess_email(email) for email in all_emails]

###########################################################################################################################################

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Use probabilities to apply a custom threshold for spam detection
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of being spam
spam_threshold = 0.000000000001 # Adjusted threshold to reduce false positives

# Predict using the custom threshold
y_pred = (y_prob >= spam_threshold).astype(int)

# Evaluate the classifier with custom threshold
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

###########################################################################################################################################

# Compute the confusion matrix with the adjusted threshold
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix with Adjusted Threshold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
