import matplotlib.dates as mdates

# convert from twitter json to datetime.date
def date_from_str(string):
    months=dict(Jan=1, Feb=2, Mar=3, Apr=4, May=5, Jun=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
    year = int(string[-4:])
    month, day = string.split(' ')[1:3]
    month = months[month]
    day = int(day)
    return datetime.date(year, month, day)

# hacky starting point -- for each topic, print p5 & p95 dates
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

# boxplot of topic distributions over time
def plot_over_time(doc_topic, topic_terms, topic_wts, df, wt_thresh=.6, sort_date=True):
    # get dates for docs in topics
    topic_dates = []
    for topic in range(doc_topic.shape[1]):
        # get list of dates
        topic_doc_wts = doc_topic[:,topic]
        # cutoff by #
        #doc_ids = np.argsort(topic_doc_wts)[-1*n_docs:]
        # cutoff by weight
        doc_ids = np.where(np.greater(topic_doc_wts, wt_thresh))[0]
        dates = tw_df.created_at.iloc[doc_ids]
        dates = dates.apply(lambda x: mdates.date2num(date_from_str(x)))
        topic_dates.append(dates)
    # make labels
    topic_labels = []
    for topic, termlist in topic_terms:
        top = list(zip(*termlist))[0][:3]
        topic_labels.append(f"{str(topic)}: " + " ".join(top))
    # sort
    if sort_date:
        sort_inds = np.argsort([d.quantile(q=.25, interpolation='nearest') for d in topic_dates])
    else:
        sort_inds = np.argsort(topic_wts) # ascending order => ascending along axis
    topic_dates = [topic_dates[i] for i in sort_inds]
    topic_labels = [topic_labels[i] for i in sort_inds]
    # plot boxplots
    plt.close()
    plt.figure(figsize=(12,8))
    plt.boxplot(topic_dates, labels=topic_labels, vert=False)
    plt.tight_layout()
    loc,_ = plt.xticks()
    lab = [mdates.num2date(d).date() for d in loc]
    plt.xticks(loc, lab)
    #plt.show()
    return plt.gcf()

