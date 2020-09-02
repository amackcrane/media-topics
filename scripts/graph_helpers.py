
import numpy as np

# Note 'cv' must have analyzer=lambda x: x; it defaults to tokenizing/lemmatizing etc. itself
def get_matrix(handles, user_text, cv):
    word_bags = [user_text[h] for h in handles]
    user_word = cv.transform(word_bags)
    return user_word

# list of handles w/ more than 'min_ct' tweets in dataset
def get_recurring_handles(tw_df, min_ct):
    rec_handles = pd.DataFrame({'count': tw_df.handle.value_counts()})
    rec_handles['handle'] = rec_handles.index
    rec_handles = rec_handles.query(f'count >= {min_ct}').handle.values
    return rec_handles

# get handle: wordlist dict
# see topic_helpers::get_unigram_lists for sense of 'binary'
def word_bags_by_handle(unigrams, tw_df, binary):
    user_text = {}
    for i in range(tw_df.shape[0]):
        handle = tw_df["handle"].iloc[i]
        text = unigrams[i].copy() # don't save the object reference!
        if not handle in user_text:
            user_text[handle] = text
        else:
            if binary:
                user_text[handle].update(text)
            else:
                user_text[handle].extend(text)
    return user_text

