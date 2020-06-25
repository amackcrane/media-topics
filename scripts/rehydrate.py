
import time
import random

import twarc

import pandas as pd


# draw in tweet IDs and sample
proportion = .0002
# random filtering takes a couple minutes; 'nrows' is fast
twt_ids = pd.read_table("data/full_dataset_clean.tsv",
                        skiprows=lambda x: False if x==0 else random.random() > proportion)
n = len(twt_ids)

print("rehyrating {} tweets".format(n))

# timer context manager (for blocks)
class Timer():
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.time = time.time() - self.start

# timer decorator (for functions)
def time_this(fun):
    def timed_function(*args):
        start = time.time()
        result = fun(*args)
        t = time.time() - start
        print("function {} with arguments {} ran in {} seconds".format(fun.__name__,
                                                                       args,
                                                                       t))
        return result
    return timed_function

# twarc it
tw = twarc.Twarc()

tweets = tw.hydrate(twt_ids.tweet_id)

fields = ["id", "lang", "full_text", "favorite_count", "in_reply_to_status_id", "in_reply_to_user_id", "in_reply_to_screen_name", "retweeted", "retweet_count", "created_at"]
user_fields = ["id", "followers_count", "friends_count", "screen_name"]
place_fields = ["name", "full_name", "country"]

tw_dicts = []

# hydrate
with Timer() as timer:
    for tweet in tweets:
        if tweet["lang"] != "en":
            continue
        ft = {k: tweet[k] for k in fields}
        ft = dict(ft, **{"user_" + k: tweet["user"][k] for k in user_fields})
        try:
            ft = dict(ft, **{"place_" + k: tweet["place"][k] for k in place_fields})
        except:
            pass
        tw_dicts.append(ft)

        # peek at geo data
        #print("geo: {}, coord: {}, place: {}, user.loc: {}".format(tweet["geo"], tweet["coordinates"], tweet["place"], tweet["user"]["location"]))

n_en = len(tw_dicts)

print("got {} English tweets from {} total; took {} seconds".format(n_en, n, timer.time))

tw_df = pd.DataFrame(tw_dicts)

tw_df.to_csv("data/loaded{}.csv".format(proportion), index=False)
