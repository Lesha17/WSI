import pandas
import torch
import numpy
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


def cluster_hidden(clusterer, vectors_or_distances):
    if clusterer.n_clusters > vectors_or_distances.shape[0]:
        raise AssertionError(f'Num clusters {clusterer.n_clusters} is greater than num words {vectors_or_distances.shape[0]}')
    clustered_labels = clusterer.fit_predict(vectors_or_distances)
    return clustered_labels


def make_labeling(datareader: BaseDataReader, bert_out,
                  word_vector_fn=vectorizing.get_avg_word_tokens_vector, bert_layer: int = -1,
                  dist_metric='cosine',
                  clustering_alg: callable = AgglomerativeClustering, num_clusters: int = 2):
    '''
    Labels every word in given data and returns a pandas series with labels

    :return: a series with labels
    '''

    df = datareader.get_dataframe()
    result = pandas.Series([-1] * len(datareader), index=df.index)

    dataset = datareader.create_dataset()
    word_vectors = word_vector_fn(bert_out, dataset, bert_layer=bert_layer)
    word_vector_norms = torch.norm(word_vectors, dim=-1)
    nonzero_words = word_vector_norms > 0

    for word in datareader.get_words():
        word_mask = datareader.get_word_df_mask(word) & nonzero_words
        num_words = numpy.sum(word_mask)
        if num_words < num_clusters:
            print(f'Num clusters {num_clusters} is greater than num words {num_words} for {word}')
            continue

        clusterer = clustering_alg(n_clusters=num_clusters, affinity=dist_metric, linkage='average')

        labels = cluster_hidden(clusterer, word_vectors[word_mask])
        result[word_mask] = labels


    return result
