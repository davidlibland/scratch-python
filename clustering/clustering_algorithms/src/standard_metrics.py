from collections import Counter
from functools import lru_cache
from itertools import combinations

from sklearn import metrics
from scipy import stats


def to_binary(clustering_list):
    return [x == y for x, y in combinations(clustering_list, 2)]


def from_binary_metric(metric):
    def clustering_metric(y_true, y_pred):
        _y_true = to_binary(y_true)
        _y_pred = to_binary(y_pred)
        return metric(_y_true, _y_pred)
    return clustering_metric


@lru_cache()
def clustering_entropy(labels: tuple):
    c = Counter(labels)
    probs = [v/len(labels) for v in c.values()]
    return stats.entropy(probs)


@lru_cache()
def cached_mutual_info(y_true: tuple, y_pred: tuple):
    return metrics.mutual_info_score(y_true, y_pred)


def normalized_mutual_information(beta):
    def computation(y_true, y_pred):
        s_y_true = tuple(sorted(y_true))
        s_y_pred = tuple(sorted(y_pred))
        true_entropy = clustering_entropy(s_y_true)
        pred_entropy = clustering_entropy(s_y_pred)
        denom = beta * true_entropy \
                + (1-beta) * pred_entropy
        if denom == 0:
            # At least one distribution has zero entropy (zero information):
            return 0
        return cached_mutual_info(tuple(y_true), tuple(y_pred))/denom
    return computation


standard_metrics = {
    "Adjusted Rand Index": metrics.adjusted_rand_score,
    "Adjusted Mutual Information": metrics.adjusted_mutual_info_score,
    "Mutual Information": metrics.mutual_info_score,
    "NMI_0": normalized_mutual_information(0.01),
    "NMI_0.25": normalized_mutual_information(0.25),
    "NMI_0.5": normalized_mutual_information(0.5),
    "Normalized Mutual Information": metrics.normalized_mutual_info_score,
    "NMI_0.75": normalized_mutual_information(0.75),
    "NMI_1": normalized_mutual_information(1),
    "Homogeneity": metrics.homogeneity_score,
    "Completeness": metrics.completeness_score,
    "V-Measure": metrics.homogeneity_completeness_v_measure,
    "Fowlkes-Mallows Score": metrics.fowlkes_mallows_score,
    "Precision": from_binary_metric(lambda true, pred: metrics.precision_recall_fscore_support(true, pred, average="binary")[0]),
    "F_0.5": from_binary_metric(lambda true, pred: metrics.fbeta_score(true, pred, 0.5)),
    "Jaccard Index": from_binary_metric(metrics.f1_score),
    "F_2": from_binary_metric(lambda true, pred: metrics.fbeta_score(true, pred, 2)),
    "F_4": from_binary_metric(lambda true, pred: metrics.fbeta_score(true, pred, 4)),
    "F_8": from_binary_metric(lambda true, pred: metrics.fbeta_score(true, pred, 8)),
    "F_16": from_binary_metric(lambda true, pred: metrics.fbeta_score(true, pred, 16)),
    "Recall": from_binary_metric(lambda true, pred: metrics.precision_recall_fscore_support(true, pred, average="binary")[1]),
}

base_metrics = {
    k: v for k, v in standard_metrics.items()
    if k in ["NMI_0", "NMI_0.25", "NMI_0.5", "NMI_0.75", "NMI_1"]
}
