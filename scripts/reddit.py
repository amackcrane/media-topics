

import praw

import pandas as pd




red = praw.Reddit()

politics = red.subreddit("politics")

query="COVID"
sort="new"
recent_pol_covid = politics.search(query=query, sort=sort)

comments = []
submissions = []

for submission in recent_pol_covid:
    # expand MoreComments
    residual=[1]
    while len(residual)>0:
        residual = submission.comments.replace_more()
    submissions.append({"id": submission.id, "title": submission.title})
    comments.extend(submission.comments.list())



fields = ["author", "body", "downs", "ups", "depth", "created", "name", "parent_id"]

cdicts = []
for comment in iter(comments):
    # skip deleted
    if not comment.author:
        continue
    fulldict = comment.__dict__
    cdict = {k: fulldict[k] for k in fields}
    cdict = dict(cdict, author=fulldict['author'].name,
                 submission=fulldict['_submission'].id)
    cdicts.append(cdict)

red_df = pd.DataFrame(cdicts)

red_df.to_csv("data/reddit.politics.{}.{}.csv".format(query, sort), index=False)
