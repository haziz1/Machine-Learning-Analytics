import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle
import os


os.chdir("C:/Users/Hassan Aziz/OneDrive/Documents/Tulane Academics/Semester 2/MGSC 7650-01-TT-Yatish/Class-Exercise/Week 7")


st.header("My sentiment model")
df = pd.read_csv("IMDB_movie_reviews_labeled.csv")

st.subheader("Training Data sample")

st.dataframe(df.sample(5))

st.write(df.sentiment.value_counts())

if st.button("Build My ML Pipeline"):
    X = df.loc[:,['review']]
    y = df.sentiment
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y)
    X_train_docs = [doc for doc in X_train.review]
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english',max_features=1000)),('cls', LinearSVC())])
    pipeline.fit(X_train_docs, y_train)

    st.subheader("Model Performance")
    #training
    training_accuracy = cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()
    st.write("training accuracy", training_accuracy)
    #validation
    predicted = pipeline.predict([doc for doc in X_test.review])
    validation_accuracy = accuracy_score(y_test, predicted)
    st.write("Validation Accuracy",validation_accuracy)
    with open('pipeline.pkl','wb') as f:
        pickle.dump(pipeline,f)

st.subheader("Testing the model")

review_text = st.text_area("Movie Review")

if st.button("Predict"):
    with open('pipeline.pkl','rb') as f:
        pipeline = pickle.load(f)
        sentiment = pipeline.predict([review_text])
        st.write("Predicted sentiment is:",sentiment[0])

st.subheader("Titanic Model Testing Form Example")
pclass = st.selectbox('pclass',options=['1','2','3'])
age = st.text_input('age')
if st.button("Predictive Survival"):
    st.write("Model Predicts")