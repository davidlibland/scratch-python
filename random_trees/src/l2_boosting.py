from typing import Callable

import numpy as np

from src.abstract_learner import AbstractLearner
from src.random_tree import MSETree


class L2Boost(AbstractLearner):
    def __init__(self, learner_factory: Callable[[], AbstractLearner],
                 num_learners: int=10,
                 step_size: float=3e-1):
        self._learner_factory = learner_factory
        self._num_learners = num_learners
        self._step_size = step_size

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._models = []
        residuals = y
        weight = 1
        for _ in range(self._num_learners):
            learner = self._learner_factory()
            learner.fit(X, residuals)
            self._models.append((weight, learner))
            # update params:
            residuals = y - self.predict(X)
            weight = self._step_size


    def predict(self, X: np.ndarray) -> np.ndarray:
        partial_results = []
        for weight, model in self._models:
            partial_results.append(weight*model.predict(X))
        return sum(partial_results)


def test_l2_boosted_tree(depth=3, num_learners=10, step_size=3e-1, irred_error=1):
    a = 3
    b = 1
    N = 1000
    X = np.random.uniform(-50, 50, (N,1))
    y = a*X+b+np.random.normal((N, 1))*irred_error

    boosted_tree = L2Boost(lambda :MSETree(depth=depth),
                           num_learners=num_learners,
                           step_size=step_size)
    boosted_tree.fit(X, y)

    # test set:
    n = 100
    X_test = np.random.uniform(-50, 50, (n, 1))
    y_test = a*X_test+b+np.random.normal((n, 1))

    # boosted pred:
    y_pred = boosted_tree.predict(X_test)

    # lin_reg:
    from sklearn.linear_model import LinearRegression
    lin_model = LinearRegression()
    lin_model.fit(X, y)
    y_lin_pred = lin_model.predict(X_test)

    print("L2Boost Error: %s, Lin Reg Error: %s" % (
        ((y_pred-y_test)**2).mean(),
        ((y_lin_pred-y_test)**2).mean()
    ))

