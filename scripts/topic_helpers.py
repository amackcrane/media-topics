

####################### Lemmatization ########################

# _.to_terms_list helper from textacy
def lower_lemma(token):
    return token.lemma_.lower()

def is_stop(tok):
    return (tok.is_stop or tok.is_punct or tok.is_space or
            lower_lemma(tok) in ["election2016", "rt"])


# originally created (I suspect?) to enforce binary term frequencies; now permits both
def get_unigram_lists(corpus, binary=False):
    if binary:
        return [{lower_lemma(tok) for tok in doc if not is_stop(tok)}
                for doc in corpus]
    else:
        return [[lower_lemma(tok) for tok in doc if not is_stop(tok)]
                for doc in corpus]


###################### Topic Visualization ##########################
    
# in: lists of top terms by topic
# out: lists side by side in data.frame for nice(r) report printing
def print_topics(topics, n_terms=30):
    # start w/ list of (topic, termlist), where termlist is list of (term, weight)
    df_dict = {}
    for top,termlist in topics:
        df_dict[top] = pd.DataFrame(termlist, columns=['word','weight'])
        #df_dict[top] = df_dict[top].iloc[0:n_terms,:]
        df_dict[top] = df_dict[top].query('weight > 1')
    df_concat = pd.concat(df_dict, axis=1, names=['topic'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_concat)


# Print 'n' top tweets for a given topic
def topic_tweets(topic, doc_topic, tw_df, n=10):
    # get col of doc-topic
    doc_weights = doc_topic[:,topic]
    # get top docs
    top_docs = np.argsort(doc_weights)[-1*n:]
    # grab text by index in df
    tweets = tw_df.full_text.iloc[top_docs]
    with pd.option_context('display.max_colwidth', 160):
        print(tweets)


# print list of terms assigned to topic
def print_topic_terms(topics):
    for top, termlist in topics:
        print(top)
        terms = pd.DataFrame(termlist, columns=['word','weight'])
        terms = terms.query('weight > 1')
        print(f"\t{terms.word.values}")

        
