import pandas as pd
import pickle
import streamlit as st


df = pd.read_csv('reddit_posts.csv')

with open('pipeline.pkl','rb') as f:
    pipeline = pickle.load(f)



df = df.loc[df.selftext.notna(),:]

st.dataframe(df.sample(5))

test_docs = list(df.selftext)

predictions = pipeline.predict(test_docs)

st.write(predictions)