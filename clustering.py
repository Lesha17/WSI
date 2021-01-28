import pandas
import torch
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering

import vectorizing
from data_readers import BaseDataReader


def calculate_distances(word_vectors, dist_metric):
    p = 2
    if dist_metric == 'l1':
        dist_metric = 'minkowski'
        p = 1
    elif dist_metric == 'l2':
        dist_metric = 'minkowski'
        p = 2
    elif dist_metric == 'manhattan':
        dist_metric = 'cityblock'

    distances = squareform(pdist(word_vectors, metric=dist_metric, p=p))
    return distances


def cluster_hidden(clusterer, distances):
    if clusterer.n_clusters > distances.shape[0]:
        raise AssertionError(f'Num clusters {clusterer.n_clusters} is greater than num words {distances.shape[0]}')
    clustered_labels = clusterer.fit_predict(distances)
    return clustered_labels


def make_labeling(datareader: BaseDataReader, bert_out,
                  word_vector_fn=vectorizing.get_word_vector_avg, bert_layer: int = -1,
                  dist_metric='cosine',
                  clustering_alg: callable = AgglomerativeClustering, num_clusters: int = 2):
    '''
    Labels every word in given data and returns a pandas series with labels

    :return: a series with labels
    '''

    result = pandas.Series([None] * len(datareader), index=datareader.get_dataframe().index)

    dataset = datareader.create_dataset()
    given_word_mask = torch.stack([smpl['given_word_mask'] for smpl in dataset])
    word_vectors = word_vector_fn(bert_out, given_word_mask, bert_layer=bert_layer)

    for word in datareader.get_words():
        word_index = datareader.get_word_df_index(word)

        distances = calculate_distances(word_vectors.loc[word_index], dist_metric)
        clusterer = clustering_alg(n_clusters=num_clusters, affinity='precomputed', linkage='average')

        labels = cluster_hidden(clusterer, distances)
        result.loc[word_index] = labels

    return result
