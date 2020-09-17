
from sklearn.feature_extraction.text import TfidfVectorizer
from textacy import Corpus
from textacy.vsm.vectorizers import Vectorizer
from textacy.tm.topic_model import TopicModel
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import datetime
import os


# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"

if not os.path.exists(f"{path}visualization"):
    os.mkdir(f"{path}visualization")

# helper functions
exec(open(f"{path}scripts/topic_helpers.py", 'r').read())


# draw in dataframe
df_path = f"{path}data/loaded_0.0002.csv"
tw_df = pd.read_csv(df_path)
print("df loaded")

corpus_path = f"{path}data/.covid_0.0002.spacy"
cache_path = f"{path}data/.covid_0.0002.unigrams.npy"

# load corpus into textacy format
if (os.path.exists(corpus_path) and (os.path.getmtime(corpus_path) >
                                     os.path.getmtime(df_path))):
    pass
else:
    chunk_size = 10_000
    tweets = Corpus("en_core_web_sm")
    for i in range((tw_df.shape[0] // chunk_size) + 1):
        tweets.add(tw_df.full_text.iloc[chunk_size * i : chunk_size * (i+1)])
        print("Parsed " + str(chunk_size * (i+1)) + " of " + str(tw_df.shape[0]))
    tweets.save(corpus_path)

binary=False

# lemmatize tweets
if (os.path.exists(cache_path) and
    (os.path.getmtime(cache_path) > os.path.getmtime(corpus_path))):        
    word_bags = np.load(cache_path, allow_pickle=True)
    print("Tweets loaded from file")
else:
    tweets = Corpus.load("en_core_web_sm", corpus_path)
    print("Tweets loaded from spacy")
    # Vectorize
    word_bags = get_unigram_lists(tweets, binary)
    np.save(cache_path, word_bags)


# vectorize

vectorizer = Vectorizer(apply_idf=True, min_df=30, max_df=.3)

doc_term = vectorizer.fit_transform(word_bags)

model = TopicModel('lda', n_topics=50)

# hmmm takes a few seconds even on 9k x 600 doc-term matrix
model.fit(doc_term)
doc_topic = model.transform(doc_term)
topics = model.top_topic_terms(vectorizer.id_to_term, top_n=10, weights=True)

# in: lists of top terms by topic
# out: lists side by side in data.frame for nice(r) report printing
def print_topics(topics, n_terms=30):
    # start w/ list of (topic, termlist), where termlist is list of (term, weight)
    df_dict = {}
    for top,termlist in topics:
        df_dict[top] = pd.DataFrame(termlist, columns=['word','weight'])
        df_dict[top] = df_dict[top].iloc[0:n_terms,:]
    df_concat = pd.concat(df_dict, axis=1, names=['topic'])
    print(df_concat)

print_topics(topics)



# hacky hacky visualization of topics over time

def date_from_str(string):
    months=dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
    year = int(string[-4:])
    month, day = string.split(' ')[1:3]
    month = months[month]
    day = int(day)
    return datetime.date(year, month, day)

def over_time(doc_topic, df, n_docs=100):
    topic_dates = []
    for topic in range(doc_topic.shape[1]):
        # get list of dates
        topic_wts = doc_topic[:,topic]
        doc_ids = np.argsort(topic_wts)[-1*n_docs:]
        dates = tw_df.created_at.iloc[doc_ids]
        dates = dates.apply(date_from_str)
        topic_dates.append(dates)
    # print p5, p95?
    for t in range(len(topic_dates)):
        print(f"Topic {t}:")
        dates = topic_dates[t]
        p5 = dates.quantile(q=.05, interpolation='nearest')
        p95 = dates.quantile(q=.95, interpolation='nearest')
        print(p5)
        print(p95)



