
from sklearn.cluster import SpectralBiclustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class SpectBiclustUser():
    """hold data, model, and visualization methods for SpectralBiclustering on twitter user-term matrix
    
    parameters:
    matrix: user-term matrix
    cluster_args: optional arguments for SpectralBiclustering model, as dict
    vectorizer: sklearn.feature_extraction.text vectorizer
    df: DataFrame w/ 'text', 'handle', and 'description' fields
    handles: list/array of document identifiers
    """
    def __init__(self, matrix, cluster_args, vectorizer, df, handles):
        self.model = SpectralBiclustering(**cluster_args)
        self.matrix = matrix
        self.vectorizer = vectorizer
        self.df = df
        self.handles = handles

    def fit(self):
        self.model.fit(self.matrix)

    # works for bi- or co-clustering!
    # but hardly shows up for sparse data
    def draw_matrix(self, filename, lognorm=False):
        """Draw heatmap over data matrix sorted by cluster.

        params:
        lognorm: log & center data first as in SpectralBiclustering(method='log')"""
        if lognorm:
            data = lognormalize(self.matrix)
        # sort
        data = data[np.argsort(self.model.row_labels_)]
        data = data[:,np.argsort(self.model.column_labels_)]
        try:
            plt.matshow(data, cmap=plt.cm.Blues)
        except ValueError:
            plt.matshow(data.todense(), cmap=plt.cm.Blues)
        plt.savefig(filename, dpi=600)


    # best to call w/ axis from plt.subplots(tight_layout=True)
    def draw_image_matrix(self, filename, lognorm=False, percentile=False):
        """Draw heatmap over a reduced matrix where rows are row clusters and columns column clusters. Cells are shaded based on average values within that cluster x cluster block.

        params:
        lognorm: log & center data first as in SpectralBiclustering(method='log')
        percentile: instead of average, color based on some percentile of block values
        """
        image, counts = self.get_image_matrix(lognorm, percentile)
        #image = image.transpose()
        _,ax = plt.subplots(tight_layout=True)
        ax.matshow(image, cmap=plt.cm.Blues)
        # set tick labels
        yticks = np.array(range(image.shape[0]))
        ylabels = list(map(lambda x: "\n".join(x),
                           self.get_handles_by_cluster(3, descriptions=True)))
        try:
            # artist-style
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, size=3)
        except AttributeError:
            # scripting-style
            ax.yticks(yticks, labels=[])
        xticks = np.array(range(image.shape[1]))
        if cv:
            xlabels = list(map(lambda x: "\n".join(x), self.get_terms_by_cluster(3)))
        else:
            xlabels=None
        try:
            # scripting-style
            ax.xticks(ticks=xticks, labels=xlabels, size=4)
        except AttributeError:
            # artist-style
            ax.set_xticks(xticks)
            ax.set_xticklabels(xlabels, rotation=90, size=4)
        # annotate w/ counts
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                # don't transpose 'counts' b/c plt.matshow orients axes funny
                ax.annotate(counts[i,j], (j,i), size=3, ha='center')
        plt.savefig(filename, dpi=500)

    def get_image_matrix(self, lognorm=False, percentile=False):
        if lognorm:
            data = lognormalize(self.matrix)
        else:
            data = self.matrix
        clusters = [pd.unique(self.model.row_labels_), pd.unique(self.model.column_labels_)]
        dim = list(map(lambda x: len(x), clusters))
        image = np.zeros(shape=dim)
        counts = np.full(shape=dim, fill_value='', dtype=object)
        for i in clusters[0]:
            for j in clusters[1]:
                submat = self.get_bicluster_submatrix(i, j)
                if percentile is False:
                    image[i,j] = np.mean(submat)
                else:
                    image[i,j] = np.percentile(submat, percentile)
                counts[i,j] = f"{submat.shape[0]}x{submat.shape[1]}"
        return image, counts

    def get_bicluster_submatrix(self, i, j):
        rows = np.where(np.equal(self.model.row_labels_, i))[0]
        columns = np.where(np.equal(self.model.column_labels_, j))[0]
        return self.matrix[rows][:,columns]



    def print_by_cluster(self, n_terms, words=True):
        if words:
            top_terms = self.get_terms_by_cluster(n_terms)
        else:
            top_terms = self.get_handles_by_cluster(n_terms)
        for i in range(len(top_terms)):
            print(i)
            # loop thru handles
            for h in top_terms[i]:
                # print in full
                desc = self.get_handle_description(h, 1000)
                desc = "; ".join(desc.split("\n"))
                print(f"   {h} -- {desc}")

    def get_terms_by_cluster(self, n_terms=5):
        """Get list of top terms for each term/column cluster"""
        # get cluster indices
        col_clusters = pd.unique(self.model.column_labels_)
        col_clusters = np.sort(col_clusters)
        # get term frequencies
        freq = np.sum(self.matrix, axis=0)
        # grr matrices
        if len(freq.shape)>1:
            freq = np.array(freq)
            freq = freq[0]
        # get int->string vocabulary
        words = self.vectorizer.get_feature_names()
        top_terms = []
        for c in col_clusters:
            # get term indices
            term_inds = np.where(np.equal(self.model.column_labels_, c))[0]
            # get frequencies
            term_freqs = freq[term_inds]
            # get top frequencies
            top_inds = term_inds[np.argsort(term_freqs)[-1*n_terms:]]
            top_cluster_terms = [words[i] for i in top_inds]
            top_cluster_terms.reverse()
            top_terms.append(top_cluster_terms)
        return top_terms


    def get_handles_by_cluster(self, n, descriptions=False):
        """Get list of top terms for each user/row cluster. 'descriptions=True' prints descriptions in lieu of handles."""
        # get cluster indices
        row_clusters = pd.unique(self.model.row_labels_)
        row_clusters = np.sort(row_clusters)
        # get usage frequencies
        freq = np.sum(self.matrix, axis=1)
        # grr matrices
        if len(freq.shape)>1:
            freq = np.array(freq)
            freq = freq[:,0]
        # get int->string vocabulary (here it's rec_handles)
        top_handles = []
        for c in row_clusters:
            # get term indices
            handle_inds = np.where(np.equal(self.model.row_labels_, c))[0]
            # get frequencies
            handle_freqs = freq[handle_inds]
            # get top frequencies
            top_inds = handle_inds[np.argsort(handle_freqs)[-1*n:]]
            if descriptions:
                top_cluster_handles = [self.get_handle_description(rec_handles[i])
                                       for i in top_inds]
            else:
                top_cluster_handles = [self.handles[i] for i in top_inds]
            top_cluster_handles.reverse()
            top_handles.append(top_cluster_handles)
        return top_handles

    def get_handle_description(self, handle, nchar=24):
        description = self.df.query('handle == @handle').iloc[0,:].description
        if pd.isna(description):
            description = str(description)
        else:
            description = description[:nchar]
        return description


    # 'bags' should be list of lemmatized tweets, not aggregated by handle
    def print_tweets_by_block(self, row, col, n, word_bags):
        """Print a sample of tweets corresponding to an intersection of row and column clusters -- i.e. a block in the image matrix.

        params:

        row, col: cluster indices
        n: # tweets to print
        word_bags: list of lemmatized tweets (not aggregated by handle)
        """
        if word_bags is None:
            # TODO could get lemmas from dataframe...
            pass
        # get row/col indices from clusterer
        rows = np.where(np.equal(self.model.row_labels_, row))[0]
        cols = np.where(np.equal(self.model.column_labels_, col))[0]
        # resolve row indices to users via rec_handles
        users = [rec_handles[i] for i in rows]
        # get df row indices from users
        df_user_inds = np.where(np.isin(self.df.handle.values, users))[0]
        # resolve col indices to lemmas from vectorizer
        vocabulary = self.vectorizer.get_feature_names()
        lemmas = np.array([vocabulary[i] for i in cols], dtype=object)
        # sanity check
        print(f"example terms: {list(lemmas[:10])}\n")
        # get df row indices by finding lemmas in list of word bags
        df_inds = []
        for i in df_user_inds:
            b = np.array(word_bags[i], dtype=object)
            if len(np.intersect1d(b, lemmas)) > 0:
                df_inds.append(i)
        # print sample of 'n' tweets
        df_inds = np.random.choice(df_inds, size=min(n, len(df_inds)), replace=False)
        for i in df_inds:
            handle = self.df.iloc[i,:].handle
            descr = self.df.iloc[i,:].description
            print(f"@{handle} - {descr}")
            print("   " + self.df.iloc[i,:].text + "\n")



# re-create the 'lognormalization' from Kluger et al
# log(1+x) and then doubly center
def lognormalize(data):
    """Take log(1+X), then subtract out row and column means"""
    data = data.copy()
    data = np.log1p(data)
    rmean = np.mean(data, axis=1)
    cmean = np.mean(data, axis=0)
    rmean = np.broadcast_to(rmean.reshape(-1,1), data.shape)
    cmean = np.broadcast_to(cmean, data.shape)
    mean = np.broadcast_to(np.mean(data), data.shape)
    data = data - cmean - rmean + mean
    return data
    



    


    

    




