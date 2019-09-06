import numpy as np
import re
import pandas as pd
import nltk
nltk.download('stopwords')
import pickle
from nltk.corpus import stopwords

######################################### Read Data ################################################
train = pd.read_csv("train_tweets.csv")
X, y = train.tweet, train.user
documents = []
from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

#########################################TFIDF################################################
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=48119, stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(documents)

#################################Training and Testing Sets####################################
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.svm import LinearSVC
classifier = LinearSVC(random_state=0, tol=1e-5)
classifier.fit(X, y)

# y_pred = classifier.predict(X_test)

# ######################################### Evaluate ###########################################
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))

######################################### Predict ###########################################
unlabeled = pd.read_csv("test_tweets_unlabeled.csv")
X_unlabeled = unlabeled.tweet
documents_unlabeled = []
for sen in range(0, len(X_unlabeled)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X_unlabeled[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)


    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    documents_unlabeled.append(document)

X_unlabeled = tfidfconverter.fit_transform(documents_unlabeled)
y_pred = classifier.predict(X_unlabeled)

################################# export predictions ################################
df_nb = pd.DataFrame(y_pred)
df_nb.to_csv("predictionsWithRNN100000features.csv")
