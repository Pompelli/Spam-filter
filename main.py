import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_emails_from_folder(folder_path, label):
    '''
    This function loads email data from a specified folder and assigns a label to each email.
    
    Parameters:
    - folder_path (str): The path to the folder containing the email files.
    - label (int): An integer label to assign to each email in the folder, typically used to
                   indicate categories such as 'ham' (0) or 'spam' (1).
                   
    Process:
    - Iterates through each file in the given folder path.
    - For each file, verifies that it is a file (not a directory).
    - Reads the content of each file, appends it to the `emails` list, and assigns the specified 
      label to the `labels` list.
    
    Returns:
    - tuple: A tuple containing:
        - emails (list of str): List of email contents read from files in the specified folder.
        - labels (list of int): List of integer labels corresponding to each email.
    '''
    
    emails = []  # Initialize an empty list to store email contents
    labels = []  # Initialize an empty list to store labels (e.g., 0 for ham, 1 for spam)
    
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)  # Create the full path to the file
        
        # Check if the current item is a file (not a subdirectory)
        if os.path.isfile(file_path):
            # Open the file and read its content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()  # Read the content of the email
                
                # Append the content to the emails list and the label to the labels list
                emails.append(content)
                labels.append(label)
    
    # Return the list of emails and their corresponding labels
    return emails, labels



def preprocess_email(content):
    '''
    This function preprocesses the text content of an email by applying several transformations
    to standardize and simplify the text, making it more suitable for analysis or machine learning models.
    
    Steps:
    - Converts all characters to lowercase.
    - Replaces all numbers with the word 'NUMBER' to generalize numerical information.
    - Replaces URLs with the word 'URL' to standardize web addresses.
    - Removes any non-word characters (e.g., punctuation), replacing them with a space.
    
    Parameters:
    - content (str): The original text content of the email.

    Returns:
    - content (str): The preprocessed text content of the email.
    '''
    
    content = content.lower()  # Convert all text to lowercase to ensure uniformity
    content = re.sub(r'\d+', 'NUMBER', content)  # Replace all numbers with the word 'NUMBER'
    content = re.sub(r'http\S+', 'URL', content)  # Replace URLs with the word 'URL'
    content = re.sub(r'\W+', ' ', content)  # Replace any non-word characters with a space
    
    return content  # Return the preprocessed email content




# Load emails from each folder
easy_ham_emails, easy_ham_labels = load_emails_from_folder('Data/easy_ham', 0)
hard_ham_emails, hard_ham_labels = load_emails_from_folder('Data/hard_ham', 0)
spam_emails, spam_labels = load_emails_from_folder('Data/spam', 1)

# Combine all emails and labels
all_emails = easy_ham_emails + hard_ham_emails + spam_emails
all_labels = easy_ham_labels + hard_ham_labels + spam_labels


processed_emails = [preprocess_email(email) for email in all_emails]


# Vectorize using CountVectorizer with n-grams
vectorizer = CountVectorizer(ngram_range=(1, 2),stop_words='english',max_df=0.5,min_df=1,binary=True)
'''
The CountVectorizer creates a dictionary (vocabulary) of all unique words (tokens) in the text.
Counting Frequencies: It counts how often each word from the vocabulary appears in each document.


ngram_range=(1, 2): 

allows vectorizer to capture not only individual words (unigrams) but also pairs of consecutive words (bigrams)


stop_words='english':

removes common English words (like “the,” “is,” “and”) which are unimportant.
helps the model focus on more meaningful words
'''

X = vectorizer.fit_transform(processed_emails)
'''
This line performs two actions:

1. The vectorizer identifys unique words/tokens in the text and assigs each a numerical index.

2. Transforming each email into a numerical feature vector,
   where the features represent the presence or frequency of words (or other tokens)

The result X is a matrix:

Rows correspond to individual emails.
Columns correspond to the features (words/tokens).
Values indicate the word frequency

Summary:
This process converts raw text into a format suitable for machine learning.
'''

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)


# Train a Naive Bayes classifier with adjusted alpha
clf = MultinomialNB(alpha=0.001)  
'''
alpha=0.001:

alpha is a smoothing parameter    default = 1
Smoothing prevents model from overemphasizing rare words.

A smaller alpha makes the model more sensitive to rare, but spam-indicative terms.
If the value is too high, the model may overlook important information.
'''
clf.fit(X_train, y_train)

# Use probabilities to apply a custom threshold for spam detection
y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of being spam
spam_threshold = 0.9

# Predict using the custom threshold
y_pred = (y_prob >= spam_threshold).astype(int)


###########################################################################################################################################
# calculate the accuracy and print the classification report
###########################################################################################################################################
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred,digits=3))



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
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
