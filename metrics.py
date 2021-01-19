from collections import defaultdict

import numpy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


def acc(y_true, y_pred):
    y_true = y_true.astype(numpy.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = numpy.zeros((D, D), dtype=numpy.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = zip(*linear_sum_assignment(w.max() - w))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def calculate_metrics(true_labels, clustered_labels):
    return {'NMI': normalized_mutual_info_score(true_labels, clustered_labels, average_method='arithmetic'),
            'ACC': acc(true_labels, clustered_labels),
            'ARI': adjusted_rand_score(true_labels, clustered_labels)}


def calculate_metrics_on_labeled_data(gold_df, labeled_df):
    result_per_word = {}
    for word in gold_df.word.unique():
        word_index = gold_df.word == word
        result_per_word[word] = calculate_metrics(gold_df[word_index].gold_sense_id.to_numpy(),
                                                  labeled_df[word_index].predict_sense_id.to_numpy())
    metrics_values = defaultdict(list)
    for word, metrics in result_per_word.items():
        for metric, value in metrics.items():
            metrics_values[metric].append(value)
    return {metric: numpy.mean(values) for metric, values in metrics_values.items()}
