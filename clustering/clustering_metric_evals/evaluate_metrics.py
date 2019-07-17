from collections import Counter
from itertools import combinations

from sklearn import metrics
import pandas as pd
from scipy import stats

from clusterings.get_rankings_packet import get_clustering_lists, ClusterType


def to_binary(clustering_list):
    return [x == y for x, y in combinations(clustering_list, 2)]


def from_binary_metric(metric):
    def clustering_metric(y_true, y_pred):
        _y_true = to_binary(y_true)
        _y_pred = to_binary(y_pred)
        return metric(_y_true, _y_pred)
    return clustering_metric


def clustering_entropy(labels):
    c = Counter(labels)
    probs = [v/len(labels) for v in c.values()]
    return stats.entropy(probs)


def normalized_mutual_information(beta):
    def computation(y_true, y_pred):
        denom = beta*clustering_entropy(y_true) \
                +(1-beta)*clustering_entropy(y_pred)
        return metrics.mutual_info_score(y_true, y_pred)/denom
    return computation


def evaluate_metric(clustering_metric, restriction: ClusterType):
    """
    :param clustering_metric: This should take two labeled lists, and return a
        number
    :return:
    """
    eval_clusters = [
        [clustering_metric(c_l["official"], c) for c in c_l["ranked"]]
        for c_l in get_clustering_lists(restriction)
    ]
    ratios = [ranking_ratio(scores) for scores in eval_clusters]
    total_pos = sum(r[0] for r in ratios)
    total = sum(r[1] for r in ratios)
    return total_pos/total


def ranking_ratio(scores):
    num_well_ordered_pairs = 0
    total_pairs = 0
    for x, y in combinations(scores, 2):
        total_pairs += 1
        if x > y:
            num_well_ordered_pairs += 1
        if x == y:
            num_well_ordered_pairs += .5
    return num_well_ordered_pairs, total_pairs


def evaluate_metrics(clustering_metrics):
    types = [ClusterType.GMR_ONLY, ClusterType.SMALL_ONLY, ClusterType.ALL]
    df = pd.DataFrame(columns=["GMR", "small cases", "all"])
    for name, metric in clustering_metrics.items():
        accuracies = [evaluate_metric(metric, r) for r in types]
        df.loc[name,:] = accuracies
    return df


standard_metrics = {
    "Adjusted Rand Index": metrics.adjusted_rand_score,
    "Adjusted Mutual Information (geom)": lambda true, pred: metrics.adjusted_mutual_info_score(true, pred, average_method="geometric"),
    "Adjusted Mutual Information (arith)": lambda true, pred: metrics.adjusted_mutual_info_score(true, pred, average_method="arithmetic"),
    "Mutual Information": metrics.mutual_info_score,
    "NMI_0": normalized_mutual_information(0),
    "NMI_0.25": normalized_mutual_information(0.25),
    "Normalized Mutual Information (geom)": lambda true, pred: metrics.normalized_mutual_info_score(true, pred, average_method="geometric"),
    "NMI_0.5": normalized_mutual_information(0.5),
    "NMI_0.5alt": lambda true, pred: metrics.normalized_mutual_info_score(true, pred, average_method="arithmetic"),
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