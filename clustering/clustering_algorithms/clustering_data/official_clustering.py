from typing import List

import numpy as np

from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from clustering_data.get_raw_data import get_official_classifier


class OfficialClustering(AbstractClusteringAlgorithm):
    def __init__(self, f_name):
        self._f_name = f_name
        self._clustering = get_official_classifier(self._f_name)

    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        labels = [self._clustering(term) for term in terms]
        return np.array(labels)
