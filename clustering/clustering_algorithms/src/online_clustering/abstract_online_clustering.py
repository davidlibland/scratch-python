from abc import ABC, abstractmethod

import numpy as np


class OnlineClustering(ABC):
    @abstractmethod
    def add_nodes(self, locations: np.ndarray):
        """
        Add nodes at the locations indicated.

        Parameters:
            locations: An array of shape (n, num_features)
        """
        raise NotImplementedError

    @abstractmethod
    def get_labels(self) -> np.ndarray:
        """
        Return a list of cluster labels for all the nodes.

        Returns:
            An array of shape (n_samples,).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """
        The number of nodes.

        Returns:
            An integer indicating the number of nodes.
        """
        raise NotImplementedError