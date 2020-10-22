
import time
import random

import twarc
import sqlite3

import pandas as pd
import numpy as np

# some datetime conversion stuff
exec(open('scripts/time_helpers.py').read())

# timer context manager (for blocks)
# time
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

# initialize 'ids' table
def init_ids_db(conn):
    conn.cursor().execute('''CREATE TABLE ids (id TEXT PRIMARY KEY, 
                                   non_en INTEGER, 
                                   hydrated INTEGER,
                                   date INTEGER,
                                   unavailable INTEGER)''')
    conn.commit()

# optionally, get IDs already hydrated
# presuming variable names constructed below
#already_hydrated = pd.read_csv("data/loaded.csv").filter(items=['id'])


# write tweet IDs to SQL with python loop -- unfathomably slow
"""
# Set up database for tracking what's been hydrated
conn = sqlite3.connect('data/full_dataset_clean_ids.db')
c = conn.cursor()
# initialize if absent
if len(c.execute('PRAGMA table_info(ids)').fetchall()) == 0:
    init_ids_db(c)
# populate
#if len(c.execute('SELECT * FROM ids').fetchone()) == 0:
# draw in tweet IDs from file
all_ids = pd.read_table("data/full_dataset_clean.tsv")
for i in reversed(range(all_ids.shape[0])):
    print(i)
    record = all_ids.iloc[i]
    # skip if exists (SQL should just reject it)
    #if len(c.execute('SELECT id FROM ids WHERE id=?', (record.tweet_id,)).fetchall()) > 0:
    #    continue
    # Fill in some values if we alreay have some twts rehydrated
    try:
        # TODO
        hydrated = int(record.tweet_id in already_hydrated)
    except NameError:
        hydrated = 0
    c.execute('INSERT INTO ids VALUES (?,?,?,?)',
              (record.tweet_id, 0, hydrated, record.date))
conn.commit()
"""


# write tweet IDs to SQL pandas style -- fast
"""
conn = sqlite3.connect('data/full_dataset_clean.db')
all_ids = pd.read_table("data/full_dataset_clean.tsv")
# 93m rows, 280s to load
all_ids.drop(columns='time', inplace=True)
all_ids = all_ids.rename(columns={'tweet_id': 'id'})

try:
    all_ids = all_ids.merge(already_hydrated, how='left', on='id', indicator=True)
    # 715s, 385s, 685s
    all_ids['hydrated'] = np.vectorize(lambda x: int(x == 'both'))(all_ids['_merge'])
    # 40s
    all_ids.drop(columns='_merge', inplace=True)
except:
    all_ids['hydrated'] = 0
all_ids['non_en'] = 0
all_ids['unavailable'] = 0
all_ids.to_sql('ids', conn, if_exists='append', index=False, chunksize=100)
# 1031s
"""



# twarc it
tw = twarc.Twarc()
to_hydrate = 10_000_000
conn = sqlite3.connect('data/full_dataset_clean.db')
c = conn.cursor()
print("connected to db")

# get list of candidate tweets
with Timer() as timer:
    candidates = c.execute('''SELECT id FROM ids 
                              WHERE hydrated=0 AND non_en=0 AND unavailable=0
                              ORDER BY Random() LIMIT ?''', (to_hydrate,)).fetchall()
    candidates = np.array([c[0] for c in candidates])
query_time = timer.time
print(f'queried {to_hydrate} ids in {query_time} seconds')
# 10000 ids in 444s = 8m
# 1000 in 1m
# 100_000 in 67s??? huh?
#   oh, I changed the steps to convert to numpy since that first line
#   maybe the array dimensionality made things harder??
# 10_000_000 in 1140s = 20m

# shuffle for temporal representativeness
#candidates = candidates[np.random.choice(candidates, len(candidates), replace=False)]

# Fields to extract
fields = ["id", "full_text", "favorite_count", "retweet_count", "created_at"]
optional_fields = ["lang", "in_reply_to_status_id", "in_reply_to_user_id", "in_reply_to_screen_name"]
quote_tweet = "quoted_status"
retweet = "retweeted_status"
lang = "lang"
user_fields = ["id", "followers_count", "friends_count", "name", "screen_name", "location", "description"]
place_fields = ["name", "full_name", "country"]

# initialize 'tweets' table
def init_tweets_db(conn):
    conn.cursor().execute('''CREATE TABLE tweets (id TEXT PRIMARY KEY,
full_text TEXT,
favorite_count INTEGER,
retweet_count INTEGER,
created_at INTEGER,
lang TEXT,
in_reply_to_status_id TEXT,
in_reply_to_user_id TEXT,
in_reply_to_screen_name TEXT,
quote_tweet INTEGER,
retweet INTEGER,
user_id TEXT,
user_followers_count INTEGER,
user_friends_count INTEGER,
user_name TEXT,
user_screen_name TEXT,
user_location TEXT,
user_description TEXT,
place_name TEXT,
place_full_name TEXT,
place_country TEXT)''')
    conn.commit()

# convert dict to tuple for sqlite
def translate(tw):
    to_return = (str(tw['id']),
                 tw['full_text'],
                 int(tw['favorite_count']),
                 int(tw['retweet_count']), 
                 time.mktime(date_from_str(tw['created_at']).timetuple()),
                 tw['lang'],
                 tw['in_reply_to_status_id'],
                 tw['in_reply_to_user_id'], 
                 tw['in_reply_to_screen_name'],
                 int(tw['quote_tweet']),
                 int(tw['retweet']),
                 str(tw['user_id']), 
                 int(tw['user_followers_count']), 
                 int(tw['user_friends_count']),
                 tw['user_name'],
                 tw['user_screen_name'],
                 tw['user_location'],
                 tw['user_description'])
    try:
        to_return = (*to_return,
                     tw['place_name'], 
                     tw['place_full_name'], 
                     tw['place_country'])
    except:
        to_return = (*to_return, None, None, None)
    return to_return


# initialze tweets db
if len(c.execute('PRAGMA table_info(tweets)').fetchall()) == 0:
    init_tweets_db(conn)
# get iterator
tweets = tw.hydrate(candidates)
successes = []
# hydrate based on number
#tw_dicts = []
finished = False
english=0

with Timer() as timer:
    for tweet in tweets:
        to_hydrate = to_hydrate-1
        tweet_id = str(tweet["id"])
        successes.append(tweet_id)
        # Handle non-english tweets (per Twitter)
        if "lang" in tweet and tweet["lang"] not in ("en", "und"):
            c.execute('UPDATE ids SET non_en=1 WHERE id=?', (tweet_id,))
            continue
        english += 1
        # Pick out fields
        tw_dict = {k: tweet[k] for k in fields}
        tw_dict = dict(tw_dict, **{k: tweet.get(k, None) for k in optional_fields})
        tw_dict["quote_tweet"] = quote_tweet in tweet
        tw_dict["retweet"] = retweet in tweet
        tw_dict = dict(tw_dict, **{"user_" + k: tweet["user"][k] for k in user_fields})
        try:
            # location fields often missing
            tw_dict = dict(tw_dict, **{"place_" + k: tweet["place"].get(k, None) for k in place_fields})
        except:
            tw_dict = dict(tw_dict, **{"place_" + k: None for k in place_fields})
        #tw_dicts.append(ft)
        # insert in tweets DB
        tw_tup = translate(tw_dict)
        c.execute(f'INSERT INTO tweets VALUES (?{",?"*(len(tw_tup)-1)})', tw_tup)
        # Record in ids DB
        c.execute('UPDATE ids SET hydrated=1 WHERE id=?', (tweet_id,))
        # twarc queries in batches of 100, so we synchronize the slow disk-writing
        if to_hydrate % 100 == 0:
            conn.commit()
            print(to_hydrate)


conn.commit()

        # peek at geo data
        #print("geo: {}, coord: {tweet["coordinates"]}, place: {tweet["place"]}, user.loc: {tweet["user"]["location"]}".format(tweet["geo"], , , ))

print(f'queried {len(successes) + to_hydrate} ids in {query_time} seconds')
print(f"retrieved {len(successes)} tweets from {len(successes) + to_hydrate} attempted; {english} retained after language filtering; took {timer.time} seconds")
# 10s for 1000
# 790s for 100_000 / 85000 / 53000

# tidy up
successes = np.array(successes)
unavailable = np.setdiff1d(candidates, successes)
unavailable = [(x,) for x in unavailable]
c.executemany('UPDATE ids SET unavailable=1 WHERE id=?', unavailable)

conn.commit()
conn.close()
