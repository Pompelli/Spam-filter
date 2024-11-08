import os

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

import re

def preprocess_email(content):
    content = content.lower()
    content = re.sub(r'\d+', 'NUMBER', content)
    content = re.sub(r'http\S+', 'URL', content)
    content = re.sub(r'\W+', ' ', content)
    return content

# Preprocess all emails
processed_emails = [preprocess_email(email) for email in all_emails]

###########################################################################################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Vectorize
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluate the classifier
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

###########################################################################################################################################

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Berechne die Konfusionsmatrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Erstelle eine Heatmap der Konfusionsmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

