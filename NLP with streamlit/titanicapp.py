# Week 7 Exercises
# Do the following in a streamlit application.

# 1. Read a labeled dataset for building machine learning pipeline. You may use titanic dataset or movie reviews dataset.

# 2. Show summary of the dataset on the app. You can include some data exploration steps and graphs.

# 3. Build your machine learning pipeline and show the performance of model training and validation step.

# 4. Create a form for user input to predict outcomes and to demonstrate your model performance to the end users.


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import pickle



st.header("Want to know if you would survive on the Titanic?")

df = pd.read_csv('titanic_raw.csv')


st.header ('Current Dataset used for Modelling')
st.dataframe(df.sample(5))


if st.button("Build My ML Pipeline"):    
    df = df.loc[:,('pclass', 'age','sex','survived','embarked')]
    df = df.loc[df.embarked.notna()]
    X = df.drop('survived', axis = 'columns')
    y = df.survived
    num_pipeline = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('norm', StandardScaler())])
    col_transformer = ColumnTransformer([
    ('num', num_pipeline, ['pclass', 'age']),
    ('cat', OneHotEncoder(), ['sex','embarked'])])
    pipeline = Pipeline([
    ('trans', col_transformer),
    ('cls', LogisticRegression())])
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    st.write("training accuracy is : ",accuracy)
    with open('pipelinetitanic.pkl','wb') as f:
        pickle.dump(pipeline,f)

st.subheader("Lets see if you'll Survive!")

age = st.text_input('age')

st.subheader('What ticket class do you think you would have bought!')
pclass = st.selectbox('pclass',options=['1','2','3'])

st.subheader('What is your gender?')
gender = st.selectbox('Gender',options=['female','male'])

embarked = st.selectbox('embarked',options = ['S','Q','C'])

abc =pd.DataFrame([[pclass,age,gender,embarked]], columns=['pclass', 'age', 'sex', 'embarked'], index=[0])


if st.button("Predict"):
    with open('pipelinetitanic.pkl','rb') as f:
        pipeline = pickle.load(f)
        survival = pipeline.predict(abc)
        st.write("Predicted survival is:",survival[0])
