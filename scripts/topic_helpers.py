

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


    
