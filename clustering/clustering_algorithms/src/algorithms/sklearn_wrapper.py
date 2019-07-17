from collections import OrderedDict
from typing import List

import numpy as np
from sklearn.base import ClusterMixin

from clustering_data.get_embedding import get_embeddings
from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm


class SKLearnWrapper(AbstractClusteringAlgorithm):
    def __init__(self, sklearn_clusterer: ClusterMixin, *embedding_files, **metadata):
        self._clusterer = sklearn_clusterer
        self._embeddings = OrderedDict()
        for f_name in embedding_files:
            self._embeddings[f_name]=get_embeddings(f_name)
        self._metadata = metadata

    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        """
        Fits the sk-learn clustering algorithm to the data and returns the
        labels
        """
        X = self.get_features(terms)
        return self._clusterer.fit_predict(X)

    def get_features(self, terms: List[str]):
        X_list = []
        for term in terms:
            features = np.hstack([
                embedding(term) for embedding in self._embeddings.values()
            ])
            X_list.append(features)
        X = np.vstack(X_list)
        return X

    def __repr__(self):
        parameters = {
            "type": "SKLearnWrapper",
            "embeddings": self._embeddings.keys(),
            "clustering_algorithm": repr(self._clusterer),
            **self._metadata
        }
        return str(parameters)
