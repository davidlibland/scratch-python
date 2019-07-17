from typing import Optional

import numpy as np
from sklearn.neighbors.ball_tree import BallTree

from src.spatial_indexes.abstract_spatial_index import SpatialIndex


class SKLearnBallTree(SpatialIndex):
    def __init__(self, k=5):
        self._nodes = None
        self._k = k

    @property
    def nodes(self) -> np.ndarray:
        """
        The nodes in index order.

        Returns:
            An array of shape (num_nodes, num_features) in index order, or
            None if no nodes have been added.
        """
        return self._nodes

    def add_nodes(self, X: np.ndarray) -> np.ndarray:
        """
        Adds nodes to the spatial index. Returns the indices of the nodes (which
        are computed incrementally).

        Parameters:
            X: An array of shape (num_samples, num_features).

        Returns:
            An array of shape (num_samples, ) and of type int. The integers
            represent the indices of the nodes added.
        """
        if self._nodes is None:
            self._nodes = X
            return np.arange(0, self._nodes.shape[0])
        else:
            n = self._nodes.shape[0]
            self._nodes = np.concatenate([self._nodes, X], axis=0)
            return np.arange(n, self._nodes.shape[0])

    def query(self, X: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Returns the k nearest neighbors.

        Parameters:
            X: An array of shape (num_samples, num_features).
            k: The number of neighbors to return.

        Returns:
            An array of shape (num_samples, k) and of type int containing the
            indices of the k nearest nodes.
        """
        if k is None:
            k = self._k
        bt = BallTree(self.nodes, metric="euclidean")
        dist, ind = bt.query(X, k)
        return ind
