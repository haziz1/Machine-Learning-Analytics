import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics

os.getcwd()
os.chdir("C:/Users/Hassan Aziz/OneDrive/Documents/Tulane Academics/Semester 2/MGSC 7650-01-TT-Yatish/Class-Exercise/Week 6")


df = pd.read_csv("IMDB_movie_reviews_labeled.csv")

# Intro Analysis

df.head
df.isna().sum()
df.sentiment.value_counts()

# Preparing for Analysis

X = df.loc[:,['review']]
y = df.sentiment

# Test & Training datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)

# Checking for stratify
y_train.value_counts()
y.value_counts()

# Converting DF into list for analysis
X_train_docs = [doc for doc in X_train.review]

# Question -- Why add fit here - wont the function autofit training dataset only

# Tokenizing the list
vect = CountVectorizer(ngram_range=(1, 3), stop_words="english",max_features=1000).fit(X_train_docs)
X_train_features = vect.transform(X_train_docs)
feature_names = vect.get_feature_names_out()

# Checking output
print("Number of features: {}".format(len(feature_names)))
print("First 100 features:\n{}".format(feature_names[:100]))
print("Every 100th feature:\n{}".format(feature_names[::100]))
print("X_train_features:\n{}".format(repr(X_train_features)))

# Modelling

lin_svc = LinearSVC(max_iter=120000)
scores = cross_val_score(lin_svc, X_train_features, y_train, cv=5)

# Printing result
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

lin_svc.fit(X_train_features, y_train)

X_test_docs = [doc for doc in X_test.review]
X_test_features = vect.transform(X_test_docs)

y_test_pred = lin_svc.predict(X_test_features)
metrics.accuracy_score(y_test, y_test_pred)