from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class SpatialIndex(ABC):
    @property
    @abstractmethod
    def nodes(self) -> np.ndarray:
        """
        The nodes in index order.

        Returns:
            An array of shape (num_nodes, num_features) in index order.
        """
        raise NotImplementedError

    @abstractmethod
    def add_nodes(self, X: np.ndarray) -> np.ndarray:
        """
        Adds nodes to the spatial index. Returns the indices of the nodes (which
        are determined by the insertion order).

        Parameters:
            X: An array of shape (num_samples, num_features).

        Returns:
            An array of shape (num_samples, ) and of type int. The integers
            represent the indices of the nodes added.
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, X: np.ndarray, k: Optional[int]=None) -> np.ndarray:
        """
        Returns the k nearest neighbors.

        Parameters:
            X: An array of shape (num_samples, num_features).
            k: The number of neighbors to return.

        Returns:
            An array of shape (num_samples, k) and of type int containing the
            indices of the k nearest nodes.
        """
        raise NotImplementedError
