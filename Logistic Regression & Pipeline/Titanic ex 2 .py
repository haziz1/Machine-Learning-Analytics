# Exercise 2
# Load the model and pipeline files created in exercise 1, load titanic_2.csv dataset and predict who will survive and who won't. 
# Create plots to show how many passengers will survive (and will not survive) across pclass and gender. 

import pickle
import pandas as pd
import matplotlib.pyplot as plt

df =  pd.read_csv('titanic_2.csv')

pipeline2 = pickle.load(open('pipeline.pkl', 'rb'))


df['survived'] = pipeline2.predict(df)

pd.crosstab(df['pclass'], df['survived']).plot.bar()
plt.show()

pd.crosstab(df['sex'], df['survived']).plot.bar()
plt.show()