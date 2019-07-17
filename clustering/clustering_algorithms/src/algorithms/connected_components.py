import logging
from itertools import combinations
from typing import List, Tuple, Callable

import numpy as np

from src.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.utils.union_find import Partition


class ConnectedComponentClusters(AbstractClusteringAlgorithm):
    def __init__(self,
                 matcher: Callable[[List[Tuple[str, str]]], List[float]],
                 threshold: float=0.5,
                 batch_size = 512
                 ):
        """
        Cluster a set of terms by computing the connected components of a graph
        whose edges are determined by a pairwise matching algorithm.

        Parameters:
            matcher: A function which takes a list or pairs, and returns a list
                of floats, where a match is indicated by the float exceeding
                the threshold. This function is assumed to be deterministic,
                and it's values will be cached for performance gains.
            threshold: A float, the threshold for matches.
            batch_size: A maximum size for batches of pairs sent to the matcher.
        """
        self._matcher = matcher
        self.threshold = threshold
        self._batch_size = batch_size
        self._match_scores = dict()

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold

    def update_match_scores(self, terms: List[str]):
        def score_unknown(term_pair):
            term_1, term_2 = term_pair
            if (term_1, term_2) in self._match_scores:
                return False
            if (term_2, term_1) in self._match_scores:
                return False
            return True

        pairs = list(filter(score_unknown, combinations(terms, 2)))
        for batch in chop_stream(self._batch_size, pairs):
            scores = self._matcher(batch)
            for term_pair, score in zip(batch, scores):
                self._match_scores[term_pair] = score

    def get_match_score(self, term_1, term_2) -> float:
        if (term_1, term_2) in self._match_scores:
            return self._match_scores[(term_1, term_2)]
        if (term_2, term_1) in self._match_scores:
            return self._match_scores[(term_2, term_1)]
        # not cached
        logging.warning(
            "Term pair %s, %s not cached. Run `update_match_scores` on sets of "
            "terms to update the cache."
            % (term_1, term_2)
        )
        score = self._matcher([(term_1, term_2)])[0]
        self._match_scores[(term_1, term_2)] = score
        return score

    def is_match(self, term_1, term_2) -> bool:
        return self.get_match_score(term_1, term_2) > self.threshold

    def txt_fit_predict(self, terms: List[str]) -> np.ndarray:
        components = Partition(
            {term: term for term in terms}
        )
        self.update_match_scores(terms)
        num_edges = 0
        for term_1, term_2 in combinations(terms, 2):
            if self.is_match(term_1, term_2):
                components.add_edge(term_1, term_2)
                num_edges += 1
        labels = [
            components[term] for term in terms
        ]
        logging.info(
            "%d edges added, %d components, for %d terms."
            % (num_edges, len(set(labels)), len(terms))
        )
        return np.array(labels).flatten()


    def __repr__(self):
        parameters = {
            "type": "ConnectedComponentClusters",
            "matching_algorithm": repr(self._matcher),
            "threshold": self.threshold,
        }
        return str(parameters)


def chop_stream(batch_size, stream):
    current_batch = []
    for x in stream:
        current_batch.append(x)
        if len(current_batch) == batch_size:
            yield [x for x in current_batch]
            current_batch = []
    if len(current_batch) > 0:
        yield current_batch
