import logging
from enum import Enum
from typing import Optional, Callable, Tuple

import numpy as np

from src.abstract_learner import AbstractLearner
from src.linked_list import Nil, LinkedList


class BinaryDecisionTree:
    class DecisionEnum(Enum):
        LEFT = "left"
        RIGHT = "right"

    class DecisionFunction:
        def __init__(self, split_feature: int, split_pt: float):
            self._split_feature = split_feature
            self._split_pt = split_pt

        @property
        def split_feature(self):
            return self._split_feature

        @property
        def split_pt(self):
            return self._split_pt

        def __repr__(self):
            return "Split on feature %d at point %s." % \
                (self.split_feature, self.split_pt)

        def __call__(self, val: np.ndarray
                     ) -> "BinaryDecisionTree.DecisionEnum":
            if val[self._split_feature] < self._split_pt:
                return BinaryDecisionTree.DecisionEnum.LEFT
            return BinaryDecisionTree.DecisionEnum.RIGHT

    def __init__(
            self,
            left_right:
            Optional[Tuple["BinaryDecisionTree", "BinaryDecisionTree"]],
            decision_fn: Optional["BinaryDecisionTree.DecisionFunction"],
            mean: Optional[np.ndarray],
            collapsed_loss: Optional[float]=None,
            data: dict=None
    ):
        assert (left_right is not None and decision_fn) or mean is not None, \
            "Must be given options and a decision, or else a mean. \n" \
            "options: %s\ndecision: %s\nmean: %s" % \
            (left_right, decision_fn, mean)
        self._left_right = left_right
        self._decision_fn = decision_fn
        self._mean = mean
        self._collapsed_loss = collapsed_loss
        self._data = data

    def __repr__(self, indent=0):
        pfx = "\t" * indent
        if self.terminal:
            return pfx + "mean: %s%s" % \
                   (self.mean, " data: %s" % self._data if self._data else "")
        return (pfx + "Decision node:\n" +
               pfx + "Decision: %s\n" +
               pfx + "Left: \n%s\n" +
               pfx + "Right: \n%s\n") % (
            self._decision_fn,
            self.left.__repr__(indent+1),
            self.right.__repr__(indent+1)
        )

    @property
    def left(
            self
    ) -> Optional["BinaryDecisionTree"]:
        return self._left_right[0]

    @property
    def right(
            self
    ) -> Optional["BinaryDecisionTree"]:
        return self._left_right[1]

    @property
    def decision_fn(self) -> Optional["BinaryDecisionTree.DecisionFunction"]:
        return self._decision_fn

    def decision(self, val: np.ndarray) -> Optional["BinaryDecisionTree"]:
        assert not self.terminal, \
            "Can't make a decision at a terminal node, %s" % self
        if self._decision_fn(val) == BinaryDecisionTree.DecisionEnum.LEFT:
            return self.left
        return self.right

    @property
    def mean(self) -> Optional[np.ndarray]:
        assert self.terminal, \
            "Means are not stored on non-terminal nodes: %s" % self
        return self._mean

    @property
    def terminal(self) -> bool:
        return not self._left_right and not self._decision_fn

    def compute_mean(self, val: np.ndarray) -> np.ndarray:
        # Descend the tree to compute the mean.
        if not self.terminal:
            return self.decision(val).compute_mean(val)
        else:
            return self.mean

    @staticmethod
    def fit_unpruned_tree(
            X: np.ndarray,
            y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], float],
            depth: int,
            attach_data: bool=False
    ) -> "BinaryDecisionTree":
        """
        Fits an unpruned tree for the given loss.

        Args:
            X: Of shape (n, d) - n datapoints, d features
            y: Of shape (n, k) - n datapoints, k output dims
            loss: Takes (mean, target) -> float
            depth: The depth of the tree.

        Returns:
            A greedily constructed binary decision tree.
        """
        n, d = X.shape
        Xy = np.concatenate([X.copy(), y.copy()], axis=1)
        # index the data sorted by features:
        Xy_s = tuple(
            Xy.copy()[np.argsort(Xy[:, j]), :]
            for j in range(d)
        )
        # compute trivial loss:
        collapsed_loss = loss(y.mean(axis=0), y)
        return BinaryDecisionTree._fit_unpruned_tree(
            Xy_s=Xy_s, loss=loss, depth=depth, num_features=d,
            collapsed_loss=collapsed_loss,
            attach_data=attach_data
        )

    @staticmethod
    def _fit_unpruned_tree(
            Xy_s: Tuple[np.ndarray, ...],
            loss: Callable[[np.ndarray, np.ndarray], float],
            depth: int,
            num_features: int,
            collapsed_loss: float=None,
            attach_data: bool=False
    ) -> Optional["BinaryDecisionTree"]:
        """
        Fits an unpruned tree for the given loss.

        Args:
            Xy_s: a d-tuple of of X-y contcatenations, where j-th array is
                sorted by the j-th index.
            loss: Takes (mean, target) -> float
            depth: The depth of the tree.

        Returns:
            A greedily constructed binary decision tree.
        """
        logging.debug("Building tree, at depth %s" % depth)
        n, _ = Xy_s[0].shape
        mean = Xy_s[0][:, num_features:].mean(axis=0)
        assert np.isclose(collapsed_loss, loss(mean, Xy_s[0][:, num_features:])), \
        "Losses don't match up: %s vs %s, sq_diff: %s" % (
            collapsed_loss,
            loss(mean, Xy_s[0][:, num_features:]),
            np.mean((collapsed_loss - loss(mean, Xy_s[0][:, num_features:]))**2)
        )
        if depth == 0 or n < 2:
            logging.info("Reached maximal depth.")
            data = None
            if attach_data:
                data = {
                    "X": Xy_s[0][:, :num_features],
                    "y": Xy_s[0][:, num_features:]
                }
            return BinaryDecisionTree(
                left_right=None,
                decision_fn=None,
                mean=mean,
                collapsed_loss=collapsed_loss,
                data=data
            )
        best_feature = None
        best_loss = None
        best_split_pt = None
        best_left_loss, best_right_loss = None, None
        for j in range(num_features):
            X = Xy_s[j][:, j]
            y = Xy_s[j][:, num_features:]
            split_pt, left_loss, right_loss = BinaryDecisionTree\
                ._optimal_splitting(X, y, loss)
            if split_pt is None:
                # There was no optimal split in that direction
                continue
            if not best_loss or best_loss > left_loss + right_loss:
                best_feature = j
                best_loss = left_loss + right_loss
                best_left_loss, best_right_loss = left_loss, right_loss
                best_split_pt = split_pt

        # Construct the decision function:
        decision_fn = BinaryDecisionTree.DecisionFunction(
            best_feature,
            best_split_pt
        )

        # Split the data:
        if best_feature is None:
            logging.info("No optimal feature split")
            # The optimal split is trivial:
            data = None
            if attach_data:
                data = {
                    "reason": "No non trivial split for data",
                    "X": Xy_s[0][:, :num_features],
                    "y": Xy_s[0][:, num_features:]
                }
            return BinaryDecisionTree(
                left_right=None,
                decision_fn=None,
                mean=mean,
                collapsed_loss=collapsed_loss,
                data=data
            )
        left_data = tuple(
            Xy[Xy[:, best_feature] < best_split_pt, :]
            for Xy in Xy_s
        )
        right_data = tuple(
            Xy[Xy[:, best_feature] >= best_split_pt, :]
            for Xy in Xy_s
        )

        # Construct the left tree:
        left = BinaryDecisionTree._fit_unpruned_tree(
            Xy_s=left_data,
            loss=loss,
            depth=depth-1,
            num_features=num_features,
            collapsed_loss=best_left_loss,
            attach_data=attach_data
        )

        # Construct the right tree:
        right = BinaryDecisionTree._fit_unpruned_tree(
            Xy_s=right_data,
            loss=loss,
            depth=depth-1,
            num_features=num_features,
            collapsed_loss=best_right_loss,
            attach_data=attach_data
        )
        return BinaryDecisionTree(
            left_right=(left, right),
            decision_fn=decision_fn,
            collapsed_loss=collapsed_loss,
            mean=mean
        )

    @staticmethod
    def _optimal_splitting(
            X: np.ndarray,
            y: np.ndarray,
            loss: Callable[[np.ndarray, np.ndarray], float],
    ) -> Tuple[int, float, float]:
        n, _ = y.shape
        assert n >= 2
        best_loss = None
        best_split_pt = None
        best_left_loss = None
        best_right_loss = None
        for split_ix in range(1, n):
            split_l = X[split_ix-1]
            split_r = X[split_ix]
            split_pt = 0.5*(split_l+split_r)
            y_l = y[X < split_pt]
            y_r = y[X >= split_pt]
            if y_l.size == 0 or y_r.size == 0:
                continue
            # ToDo: optimize mean computation:
            mean_l = y_l.mean(axis=0)
            mean_r = y_r.mean(axis=0)
            left_loss = loss(mean_l, y_l)
            right_loss = loss(mean_r, y_r)
            cur_loss = left_loss + right_loss
            if not best_loss or best_loss > cur_loss:
                best_loss = cur_loss
                best_split_pt = split_pt
                best_left_loss, best_right_loss = left_loss, right_loss
        return best_split_pt, best_left_loss, best_right_loss

    @property
    def loss_delta_if_collapsed(self) -> float:
        assert self._collapsed_loss is not None, "Must know the collapsed loss"
        assert not self.terminal, "Already collapsed"
        return self._collapsed_loss - \
               self.left._collapsed_loss - self.right._collapsed_loss

    @property
    def total_loss(self) -> float:
        assert self._collapsed_loss is not None, "Must know the collapsed loss"
        if self.terminal:
            return self._collapsed_loss
        return self.left.total_loss + self.right.total_loss

    @property
    def height(self) -> int:
        if self.terminal:
            return 0
        return max(self.left.height, self.right.height) + 1

    def collapse(self):
        self._decision_fn = None
        self._left_right = None
        assert self.terminal

    def rev_prune_order(self) -> LinkedList["BinaryDecisionTree"]:
        """Iterate over nodes from leaves inwards with each step minimizing
        the collapse loss"""
        assert not self.terminal, "Can't prune a terminal tree"
        if not self.left.terminal:
            left_order = self.left.rev_prune_order()
        else:
            left_order = Nil
        if not self.right.terminal:
            right_order = self.right.rev_prune_order()
        else:
            right_order = Nil
        return LinkedList(
            self,
            LinkedList.merge(left_order, right_order,
                             key=lambda x: -x.loss_delta_if_collapsed),
        )

    def prune_order(self) -> LinkedList["BinaryDecisionTree"]:
        return self.rev_prune_order().reverse()


def mse_loss(mean: np.ndarray, target: np.ndarray) -> float:
    return float(np.sum((mean-target)**2))


class MSETree(AbstractLearner):
    def __init__(self, depth: int):
        self._depth = depth
        self._tree = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._tree = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss,
                                                          depth=self._depth)

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._tree is not None, "Must fit the tree before predicting."
        n = X.shape[0]
        vals = []
        for i in range(n):
            vals.append(self._tree.compute_mean(X[i,:]))
        return np.vstack(vals)
