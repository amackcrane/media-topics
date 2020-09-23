
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
exec(open(f"{path}scripts/time_helpers.py", 'r').read())


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



# Look at tweet density over time
plt.figure(figsize=(8,6))
tw_df.created_at.apply(date_from_str).hist(bins=70)
plt.xticks(rotation=60)
plt.tight_layout()
#plt.show()
#plt.savefig("visualization/date_hist")


# vectorize
vectorizer = Vectorizer(apply_idf=True, min_df=30, max_df=.3)

doc_term = vectorizer.fit_transform(word_bags)


# Topic Model
model = TopicModel('lda', n_topics=50)

# hmmm takes a few seconds even on 9k x 600 doc-term matrix
model.fit(doc_term)
doc_topic = model.transform(doc_term)
topic_terms = list(model.top_topic_terms(vectorizer.id_to_term, top_n=40, weights=True))
topic_wts = model.topic_weights(doc_topic)
# grrr generators
#topic_docs = list(model.top_topic_docs(doc_topic, weights=True))


# Inspect
# top terms by topic
print_topics(topic_terms)

# boxplot of topic distributions over time
f=plot_over_time(doc_topic, topic_terms, topic_wts, tw_df, sort_date=True)
f.savefig("visualization/box_date")

# pull up tweets for a given topic
topic_tweets(4, doc_topic, tw_df, n=30)


