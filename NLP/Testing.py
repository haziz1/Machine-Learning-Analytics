class Category:
  BOOKS = "BOOKS"
  CLOTHING = "CLOTHING"

train_x = ["i love the book", "this is a great book", "the fit is great", "i love the shoes"]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(binary =True,ngram_range = (1,2))
train_x_vectors = vectorizer.fit_transform(train_x)

vectorizer.get_feature_names_out()
train_x_vectors.toarray()

from sklearn import svm

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)

test_X = vectorizer.transform(['I like this fit'])
clf_svm.predict(test_X)

import spacy

nlp = spacy.load("en_core_web_md")

print(train_x)

docs = [nlp(text) for text in train_x]

train_x_word_vectors = [x.vector for x in docs]


clf_svm_wv = svm.SVC(kernel = 'linear')
clf_svm_wv.fit(train_x_word_vectors,train_y)


test_x = ['i wore a book']
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]
clf_svm_wv.predict(test_x_word_vectors)


# regex

import re

regexp = re.compile(r"^ab[^\s]*cd$")

phrases = ['abcd','xyz','123','xxabcd','ab123xxcd']

matches = []
for phrase in phrases:
  if re.match(regexp,phrase):
    matches.append(phrase)
print(matches)


re.match(regexp,phrases)


import nltk

# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

phrase = 'reading the books'

words = word_tokenize(phrase)

stemmer.stem(phrase)

stemmed_words = []
for word in words:
  stemmed_words.append(stemmer.stem(word))

print(stemmed_words)

" ".join(stemmed_words)


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
phrase = 'here is an example of stopwords hehe haha i am amazing'

stripped_phrase = []

words = word_tokenize(phrase)

for word in words:
  if word not in stop_words:
    stripped_phrase.append(word)

print(stripped_phrase)

