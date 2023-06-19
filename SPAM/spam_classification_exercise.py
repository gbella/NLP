import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.linear_model import LogisticRegression

# ============================================================
# Load the messages and the corresponding labels:
# 1 = spam, 0 = not spam
# ============================================================

#-----------
# Exercise 1
#-----------
data = pd.read_csv('sms_spam.csv',sep=';')
#-----------
# Exercise 2
#-----------
#data = pd.read_csv('email_spam.csv')
#--------------
# Exercises 1+2
#--------------
messages = data["message"]
labels = data["label"]
    #----------------------------------------------------------
    # Shuffle and split the data into a training and a test set
    # (80% training, 20% test). The random_state is fixed
    # so that we get the same random shuffle at every execution.
    #----------------------------------------------------------
training_messages, testing_messages, training_labels, testing_labels = train_test_split(messages, labels, test_size=0.2, random_state=42)
#--------------
# Exercises 3+4
#--------------
#sms_data = pd.read_csv('sms_spam.csv',sep=';')
#email_data = pd.read_csv('email_spam.csv')
#-----------
# Exercise 3
#-----------
#training_messages = sms_data["message"]
#training_labels = sms_data["label"]
#testing_messages = email_data["message"]
#testing_labels = email_data["label"]
#-----------
# Exercise 4
#-----------
#messages = pd.concat([email_data['message'], sms_data['message']])
#labels = pd.concat([email_data['label'], sms_data['label']])
#training_messages, testing_messages, training_labels, testing_labels = train_test_split(messages, labels, test_size=0.2, random_state=42)

# ============================================================
# CODE COMMON TO ALL EXERCISES
# ============================================================


# ============================================================
# Preprocess input text:
#  - tokenize (find word boundaries intelligently, using nltk)
#  - eliminate stop words
# ============================================================

def preprocess(text):
    text = str(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

training_messages = training_messages.apply(preprocess)
testing_messages = testing_messages.apply(preprocess)

# ============================================================
# Compute input features using TF-IDF
# ============================================================
cv = TfidfVectorizer(max_features=500)
train_features = cv.fit_transform(training_messages).toarray()
test_features = cv.transform(testing_messages).toarray()
# ============================================================
# Train the model over the training set
# ============================================================
clf = LogisticRegression(max_iter=2000)
clf.fit(train_features, training_labels)
# ============================================================
# Run the prediction over the test set
# ============================================================
predicted_labels = clf.predict(test_features)

# ============================================================
# Evaluate the model
#   truth: array containing ground truth labels
#   pred: array containing predicted labels
#   pos_label: the label that should be considered as positive
# ============================================================
# Accuracy = correct_predictions / total_predictions
def accuracy_score(truth, pred):
    # TODO: COMPLETE THE CODE
    return 0

# Precision = true_positives / (true_positives + false_positives)
def precision_score(truth, pred, pos_label):
    # TODO: COMPLETE THE CODE
    return 0
    
# Recall = true_positives / (true_positives + false_negatives)
def recall_score(truth, pred, pos_label):
    # TODO: COMPLETE THE CODE
    return 0

def f1_score(truth, pred, pos_label):
    # TODO: COMPLETE THE CODE
    return 0

print('Accuracy:', accuracy_score(testing_labels.values, predicted_labels))
print('Precision:', precision_score(testing_labels.values, predicted_labels, pos_label=1))
print('Recall:', recall_score(testing_labels.values, predicted_labels, pos_label=1))
print('F1 score:', f1_score(testing_labels.values, predicted_labels, pos_label=1))

