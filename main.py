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


def preprocess_email(content):
    content = content.lower()
    content = re.sub(r'\d+', 'NUMBER', content)
    content = re.sub(r'http\S+', 'URL', content)
    content = re.sub(r'\W+', ' ', content)
    return content



# Load emails from each folder
easy_ham_emails, easy_ham_labels = load_emails_from_folder('Data/easy_ham', 0)
hard_ham_emails, hard_ham_labels = load_emails_from_folder('Data/hard_ham', 0)
spam_emails, spam_labels = load_emails_from_folder('Data/spam', 1)

# Combine all emails and labels
all_emails = easy_ham_emails + hard_ham_emails + spam_emails
all_labels = easy_ham_labels + hard_ham_labels + spam_labels


processed_emails = [preprocess_email(email) for email in all_emails]


# Vectorize using CountVectorizer with n-grams
vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words='english')  # Unigrams und Bigrams
'''
ngram_range=(1, 2): 

allows vectorizer to capture not only individual words (unigrams) but also pairs of consecutive words (bigrams)


stop_words='english':

removes common English words (like “the,” “is,” “and”) which are unimportant.
helps the model focus on more meaningful words
'''

X = vectorizer.fit_transform(processed_emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)


# Train a Naive Bayes classifier with adjusted alpha
clf = MultinomialNB(alpha=0.3)  
'''
alpha=0.3:

alpha is a smoothing parameter    default = 1
Smoothing prevents model from overemphasizing rare words.

A smaller alpha makes the model more sensitive to rare, but spam-indicative terms.
If the value is too high, the model may overlook important information.
'''
clf.fit(X_train, y_train)

# Use probabilities to apply a custom threshold for spam detection
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of being spam
spam_threshold = 0.1 

# Predict using the custom threshold
y_pred = (y_prob >= spam_threshold).astype(int)

# Evaluate the classifier with custom threshold
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))



###########################################################################################################################################
#Plotting the results
###########################################################################################################################################


# Plot histogram of predicted probabilities for spam
plt.figure(figsize=(10, 6))
plt.hist(y_prob, bins=50, color='skyblue', edgecolor='k', alpha=0.7)
plt.axvline(spam_threshold, color='red', linestyle='--', label=f'Spam Threshold ({spam_threshold})')
plt.title("Histogram of Predicted Spam Probabilities")
plt.xlabel("Predicted Probability of Spam")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Compute the confusion matrix with the adjusted threshold
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.title("Confusion Matrix with Adjusted Threshold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
