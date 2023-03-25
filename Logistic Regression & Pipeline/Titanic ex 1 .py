import pandas as pd
import os

os.getcwd()
os.chdir("C:/Users/Hassan Aziz/OneDrive/Documents/Tulane Academics/Semester 2/MGSC 7650-01-TT-Yatish/Class-Exercise/Week 4")
df = pd.read_csv('titanic_raw.csv')

# Part 1
# Using pandas, fully explore the titanic_raw.csv dataset and prepare the dataset for machine learning 
# (eg. find shape, value counts, datatypes of columns, NA values, correlated columns, etc.) 


df.head()

df.corr(numeric_only=True)['survived'].sort_values(ascending=False)

df.shape

df.isna().sum()

df.describe()

df.dtypes

df.nunique()

# Part 2
# Build a machine learning pipeline to predict survival of passengers using the titanic_raw.csv dataset, save the model and pipeline files. 
# Print cross validation score, accuracy of your model and plot confusion matrix.

df = df.loc[:,('pclass', 'age','sex','survived','embarked')]

df.isnull().sum()

df = df.loc[df.embarked.notna()]

df_X = df.drop('survived', axis = 'columns')
df_Y = df.survived

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_pipeline = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('norm', StandardScaler())])

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

col_transformer = ColumnTransformer([
    ('num', num_pipeline, ['pclass', 'age']),
    ('cat', OneHotEncoder(), ['sex','embarked'])
])

pipeline = Pipeline([
    ('trans', col_transformer),
    ('cls', LogisticRegression())
])

from sklearn.model_selection import train_test_split, cross_val_score

x_train, x_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.3, stratify=df_Y)

pipeline.fit(x_train, y_train)

# Model Training Accuracy

scores = cross_val_score(pipeline, x_train, y_train, scoring='accuracy', cv = 5)

scores
scores.mean()

y_pred = pipeline.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

accuracy_score(y_pred, y_test)

cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
cm

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= pipeline.classes_)
plot.plot()
plt.show()

import pickle

pickle.dump(pipeline, open('pipeline.pkl', 'wb'))
pipeline2 = pickle.load(open('pipeline.pkl', 'rb'))