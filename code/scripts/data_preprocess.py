import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from textblob import Word, TextBlob
from wordcloud import WordCloud  # visualization of words
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder

nltk.download('all')

df = pd.read_csv("/home/guillaume/nlp_sentiment_detection/data/product_reviews/Books_rating.csv", 
                  nrows = 1000)
df = df.rename(columns = {"review/text" : "review"})

df["review"] = df["review"].str.lower()


lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text_url = re.sub(r'http\S+', '', text)
    stop_text = " ".join([x for x in text_url.split() if x not in stopwords.words("english")])
    punctuation_text = re.sub('[^\w\s]', '', stop_text)
    lem_words = [lemmatizer.lemmatize(token) for token in punctuation_text.split()]
    return " ".join([x for x in lem_words if x not in stopwords.words("english")])

df["review_clean"] = df["review"].map(preprocess_text)


count_words = df["review_clean"].apply(lambda x: pd.Series(x.split(" ")).value_counts()).sum(axis=0).reset_index()
count_words = count_words.rename(columns = {"index" : "word", 0 : "count"})

df = df[df["review/score"]!=3]
df["score"] = df["review/score"].map(lambda x: 1 if x>3 else 0)

words_to_keep = count_words[count_words["count"]>=10]["word"].tolist()


def keep_words(text, words_to_keep = words_to_keep):
    words_to_keep = [token for token in text.split() if token in words_to_keep]
    return " ".join(words_to_keep)
df["review_top_words"] = df["review_clean"].map(keep_words)

vectoriser = CountVectorizer()
X = vectoriser.fit_transform(df["review_top_words"])

print(X.toarray().shape)