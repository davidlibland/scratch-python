import random
from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Optional, Tuple

import numpy as np

from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from clustering_data.get_raw_data import get_ddxs, get_f_names
from clustering_data.official_clustering import OfficialClustering


class AlgorithmEvaluator:
    def __init__(self, solve_range: Optional[Tuple[int, int]]=None, **metrics):
        """
        This class encapsulates the functionality to evaluate various clustering
        metrics against various clustering algorithms.

        Parameters:
            solve_range: A pair of integers indicating a lower and upper bound
                on the number of solves.
            **metrics: A dict of clustering metrics to use. Each metric should
                take two parameters: a list of true cluster labels, and a list
                of predicted cluster labels, and return a floating point value.
        """
        self._solve_range = solve_range
        self._metrics = metrics
        self._case_limit = None

    @property
    def case_limit(self):
        return self._case_limit

    @case_limit.setter
    def case_limit(self, n):
        self._case_limit = n

    def evaluate_on(self, clustering_alg: AbstractClusteringAlgorithm,
                    f_name: str) -> Dict[str, float]:
        """
        Evaluates a clustering algorithm on a given case.

        Parameters:
            clustering_alg: A clustering algorithm to evaluate
            f_name: The name of a clustering packet file
            (e.g. "clustering_data/q_packets/q_packet_16")

        Returns:
            A dict mapping metrics to their results.
        """
        dxs = self.get_dxs(f_name)
        y_true = self.get_y_true(f_name)
        y_pred = clustering_alg.txt_fit_predict(dxs)
        scores = {key: metric(y_true, y_pred)
                  for key, metric in self._metrics.items()}
        return scores

    def evaluate(self, clustering_alg: AbstractClusteringAlgorithm
                 ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates a clustering algorithm across all cluster packets.

        Parameters:
            clustering_alg: A clustering algorithm to evaluate

        Returns:
            A dict mapping clustering packets to metrics to their results.
        """
        f_names = self.get_f_names()
        return {
            f_name: self.evaluate_on(clustering_alg, f_name)
            for f_name in f_names
        }

    def evaluate_mean(self, clustering_alg: AbstractClusteringAlgorithm
                 ) -> Dict[str, float]:
        """
        Evaluates the mean of each metric across all clustering packets.

        Parameters:
            clustering_alg:

        Returns:

        """
        f_names = self.get_f_names()
        score_cnts = defaultdict(lambda: (0, 0))
        for f_name in f_names:
            dxs = self.get_dxs(f_name)
            y_true = self.get_y_true(f_name)
            y_pred = clustering_alg.txt_fit_predict(dxs)
            for key, metric in self._metrics.items():
                try:
                    score = metric(y_true, y_pred)
                    total_score, cnt = score_cnts[key]
                    score_cnts[key] = total_score + score, cnt + 1
                except Exception as exc:
                    print("Error for metric %s on %s: %s" % (key, f_name, exc))
        result = {}
        for key, (score, cnt) in score_cnts.items():
            if cnt != 0:
                result[key] = score/cnt
            else:
                result[key] = float("nan")
        return result

    @lru_cache()
    def get_dxs(self, f_name: str) -> List[str]:
        """
        Given the file name of a clustering packet, this returns a list
        of diagnoses.
        """
        ddxs = get_ddxs(f_name)
        if self._solve_range:
            num_solves = random.randint(self._solve_range[0], self._solve_range[1])
            ddxs = ddxs[:num_solves]
        return [dx for ddx in ddxs for dx in ddx]

    @lru_cache()
    def get_y_true(self, f_name: str) -> np.ndarray:
        """
        Given a file name of a clustering packet, this returns the
        "Gold Standard" cluster labels.
        """
        dxs = self.get_dxs(f_name)
        official_clustering = OfficialClustering(f_name)
        return official_clustering.txt_fit_predict(dxs)

    @lru_cache()
    def get_f_names(self) -> List[str]:
        """
        This returns a list of all clustering packet file names.
        """
        return get_f_names(self.case_limit)
