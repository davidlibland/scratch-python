from abc import ABC, abstractmethod

import numpy as np


class AbstractLearner(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError