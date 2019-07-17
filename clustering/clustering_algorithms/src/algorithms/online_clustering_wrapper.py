from typing import List, Callable

import numpy as np
from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.online_clustering.abstract_online_clustering import OnlineClustering


class OnlineClusteringWrapper(AbstractClusteringAlgorithm):
    def __init__(self,
                 online_clusterer_factory: Callable[[], OnlineClustering],
                 embedding,
                 **metadata):
        self._clusterer_factory = online_clusterer_factory
        self._embedding = embedding
        try:
            self._embedding_fnames = self._embedding.f_names
        except:
            self._embedding_fnames = []
        self._metadata = metadata

    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        """
        Fits the online clustering algorithm to the data and returns the
        labels
        """
        X = self.get_features(terms)
        clusterer = self._clusterer_factory()
        assert X.shape[0] == len(terms), \
            "Computed features have incorrect shape."
        assert clusterer.num_nodes == 0, \
            "Clusterer factory produced stale clusterer."
        clusterer.add_nodes(X)
        assert clusterer.num_nodes == len(terms), \
            "Clustering failed to add the correct number of nodes."
        labels = clusterer.get_labels()
        return labels

    def get_features(self, terms: List[str]):
        """
        Returns features for the terms.
        """
        X_list = []
        for term in terms:
            features = self._embedding(term)
            X_list.append(features)
        X = np.vstack(X_list)
        return X

    def __repr__(self):
        parameters = {
            "type": "OnlineClusteringWrapper",
            "embedding": self._embedding_fnames,
            "clustering_algorithm": repr(self._clusterer_factory()),
            **self._metadata
        }
        return str(parameters)
