

"""compute checkerboard biclustering of tweet dataset, storing tweets in a textacy Corpus and processing using vectorization and models from scikit-learn"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import SpectralCoclustering, SpectralBiclustering
from textacy import Corpus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# fix wd
path = ""
if os.path.basename(os.getcwd()) == "scripts":
    path = "../"

if not os.path.exists(f"{path}visualization"):
    os.mkdir(f"{path}visualization")
    
# load helpers
exec(open(f"{path}scripts/topic_helpers.py", 'r').read())
exec(open(f"{path}scripts/graph_helpers.py", 'r').read())
exec(open(f"{path}scripts/bicluster_object.py", 'r').read())


    
# draw in dataframe
df_path = f"{path}data/test_data.csv"
tw_df = pd.read_csv(df_path)
print("df loaded")

corpus_path = f"{path}data/.test.spacy"
cache_path = f"{path}data/.test.unigrams.npy"

# load corpus into textacy format
if (os.path.exists(corpus_path) and (os.path.getmtime(corpus_path) >
                                     os.path.getmtime(df_path))):
    pass
else:
    chunk_size = 10_000
    tweets = Corpus("en_core_web_sm")
    for i in range((tw_df.shape[0] // chunk_size) + 1):
        tweets.add(tw_df.text.iloc[chunk_size * i : chunk_size * (i+1)])
        print("Parsed " + str(chunk_size * (i+1)) + " of " + str(tw_df.shape[0]))
    tweets.save(corpus_path)

binary=False

# lemmatize tweets
if (os.path.exists(cache_path) and
    (os.path.getmtime(cache_path) > os.path.getmtime(corpus_path))):        
    unigrams = np.load(cache_path, allow_pickle=True)
    print("Tweets loaded from file")
else:
    tweets = Corpus.load("en_core_web_sm", corpus_path)
    print("Tweets loaded from spacy")
    # Vectorize
    unigrams = get_unigram_lists(tweets, binary)
    np.save(cache_path, unigrams)

# get users w/ 3+ tweets in dataset
rec_handles = get_recurring_handles(tw_df, 3)
# get user : list of lemmas dict (aggregated over user's tweets)
user_bags = word_bags_by_handle(unigrams, tw_df, binary)


# min_df = 30 cuts down from 100k terms to 3000
# note pass vacuous 'analyzer' b/c we've already lemmatized
cv = CountVectorizer(analyzer=lambda x: x, min_df = 30)

# get doc-term matrix for all users and recurring (3+) users
doc_term = cv.fit_transform(unigrams)
doc_term_rec = get_matrix(rec_handles, user_bags, cv)


# bi-clustering
bicluster_params = {'n_clusters': (20,20), 'method': 'log', 'random_state': 10}

# Note this runs out of memory on full doc_term matrix
bispec = SpectBiclustUser(matrix=doc_term_rec.toarray(), cluster_args=bicluster_params, vectorizer=cv, df=tw_df, handles=rec_handles)

bispec.fit()
bispec.draw_image_matrix("visualization/bispec", lognorm=True)



# try a binary version
doc_term_bin = doc_term_rec.todense()
doc_term_bin = np.where(np.greater(doc_term_bin, 1), 1, doc_term_bin)
# toss in noise to avoid numerical troubles
doc_term_bin = doc_term_bin + np.random.normal(0, 1e-10, doc_term_bin.shape)

binary_spec = SpectBiclustUser(matrix=doc_term_bin, cluster_args=bicluster_params, vectorizer=cv, df=tw_df, handles=rec_handles)
binary_spec.fit()

# image matrix plot
binary_spec.draw_image_matrix("visualization/binary", lognorm=True)

# full matrix plot
binary_spec.draw_matrix("visualization/binary_full", lognorm=True)
