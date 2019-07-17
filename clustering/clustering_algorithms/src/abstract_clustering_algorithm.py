from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from typing import List, Dict

import numpy as np


class AbstractClusteringAlgorithm(ABC):
    @abstractmethod
    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        """
        Given a set of text terms, an implementation of this function should
        cluster them, and return a set of cluster labels.

        Note: Implementations need not be deterministic, or reuse the same
            cluster labels when run multiple times.

        Parameters:
            terms: The terms to cluster

        Returns:
            A corresponding list of cluster labels.
        """
        raise NotImplementedError

    def get_clustering_dict(self, terms: List[str]) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping cluster labels to clusters.

        Parameters:
            terms: The terms to cluster

        Returns:
            A dictionary of clusters
        """
        labels = self.txt_fit_predict(terms)
        pre_cluster_dict = defaultdict(list)
        for label, term in zip(labels, terms):
            pre_cluster_dict[label].append(term)
        cluster_dict = OrderedDict()
        for label, sublabels in sorted(pre_cluster_dict.items(), key=lambda kv: -len(kv[1])):
            cluster_dict[label] = sublabels
        return cluster_dict

    @staticmethod
    def save_clusters_as_tsv(cluster_dict, f_name):
        """
        Saves clusters to the specified file.

        Parameters:
            terms: The terms to cluster
        """
        formatted_lines = []
        for label, sublabels in sorted(cluster_dict.items(), key=lambda kv: -len(kv[1])):
            formatted_lines.append(label)
            for sublabel in set(sublabels):
                formatted_lines.append("\t"+sublabel)
        with open(f_name, "w") as f:
            f.write("\n".join(formatted_lines))
