from typing import List

from sklearn.base import ClusterMixin
import numpy as np

from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from clustering_data.get_embedding import get_embeddings


class SKLearnClusteringAlgorithm(AbstractClusteringAlgorithm):
    def __init__(self, clustering_algorithm: ClusterMixin):
        self._clustering_algorithm = clustering_algorithm
        self._embedding = get_embeddings()

    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        X = np.vstack([self._embedding(term) for term in terms])
        return self._clustering_algorithm.fit_predict(X)
