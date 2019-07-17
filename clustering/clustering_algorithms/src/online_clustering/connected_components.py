from typing import Callable

import numpy as np

from src.online_clustering.abstract_online_clustering import OnlineClustering
from src.utils.incremental_linkage_graph import IncrementalLinkageGraph, Node
from src.utils.union_find import Partition


class FastConnectedComponentClusters(OnlineClustering):
    def __init__(self,
                 match_scores: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 score_threshold: float=0.5,
                 num_neighbors: int=5
                 ):
        """
        Cluster a set of terms by computing the connected components of a graph
        whose edges are determined by a pairwise matching algorithm.

        Parameters:
            match_scores: A function which returns an array consisting of match
                scores: for array parameters left, right (of shapes
                (n, num_features)), the result is of shape (n,) and result[i]
                is the score associated to a match between left[i,:] and
                right[i,:].
            score_threshold: A float, the threshold above which a match is
                formed.
            num_neighbors: An int - the number of nearest neighbors to check
                for matches.
        """
        self._match_scores = match_scores
        self.score_threshold = score_threshold
        self._num_neighbors = num_neighbors
        self._clusters = Partition({})
        self._incremental_linkages = IncrementalLinkageGraph(
            link_cost=lambda X, Y: 1 - match_scores(X, Y),
            cost_threshold= 1 - score_threshold,
            num_neighbors=num_neighbors
        )

    @property
    def score_threshold(self):
        return self._score_threshold

    @score_threshold.setter
    def score_threshold(self, threshold):
        self._score_threshold = threshold

    def add_nodes(self, locations: np.ndarray):
        """
        Add nodes at the locations indicated.

        Parameters:
            locations: An array of shape (n, num_features)
        """
        current_num_nodes = self.num_nodes
        # Add to our store of nodes:
        for i in range(locations.shape[0]):
            location = locations[i,:].reshape([1,-1])
            node_id = i+current_num_nodes
            ix, new_merges = self._incremental_linkages.add_node(
                Node(location=location, data=node_id)
            )
            assert ix == node_id, \
                "Mismatched node ids %d vs %d. This is a bug." % (ix, node_id)
            self._clusters[node_id] = node_id
            for merge_node in new_merges:
                # Get the id of the node with which to merge
                m_node_id = merge_node.data
                self._clusters.add_edge(node_id, m_node_id)

    def get_labels(self) -> np.ndarray:
        """
        Return a list of cluster labels for all the nodes.

        Returns:
            An array of shape (n_samples,).
        """
        labels = [label for _, label in sorted(self._clusters.items())]
        return np.array(labels).reshape([-1])

    @property
    def num_nodes(self) -> int:
        """
        The number of nodes.

        Returns:
            An integer indicating the number of nodes.
        """
        return len(self._clusters.keys())

    def __repr__(self):
        parameters = {
            "type": "ConnectedComponentClusters",
            "matching_algorithm": repr(self._match_scores),
            "score_threshold": self.score_threshold,
            "num_neighbors": self._num_neighbors
        }
        return str(parameters)
