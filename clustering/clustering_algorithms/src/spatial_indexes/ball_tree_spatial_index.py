import random
from collections.abc import Iterator
import heapq as hq
from itertools import islice
from typing import Tuple, Any, Optional, Callable

import numpy as np

from src.spatial_indexes.abstract_spatial_index import SpatialIndex
from src.spatial_indexes.cheap_ball_tree import build_cheap_ball_tree
from src.spatial_indexes.core_ball_tree import BallTreeNode, Ball


class BallTreeSpatialIndex(SpatialIndex):
    def __init__(
            self,
            online_ball_tree_constructor:
            Optional[
                Callable[
                    [Ball, Optional[Any], Optional[BallTreeNode]],
                    BallTreeNode
                ]
            ]=None,
            projection_dim: Optional[int]=10,
    ):
        """
        A spatial index based on ball trees. This becomes faster than sklearn's
        implementation for online queries at approximately 4000 pts in 200 dims.
        
        Parameters:
            online_ball_tree_constructor: A function used to incrementally 
                construct a ball tree.
            projection_dim: Ball centers are projected onto the first 
                `projection_dim` dimensions, to help avoid the curse of 
                dimensionality (if provided).
        """
        if online_ball_tree_constructor is None:
            online_ball_tree_constructor = build_cheap_ball_tree
        self._root = None
        self._online_ball_tree_constructor = online_ball_tree_constructor
        self._nodes = None
        if projection_dim is not None:
            self._projection = lambda x: x[:projection_dim]
        else:
            self._projection = lambda x: x

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
        are determined by the insertion order).

        Parameters:
            X: An array of shape (num_samples, num_features).

        Returns:
            An array of shape (num_samples, ) and of type int. The integers
            represent the indices of the nodes added.
        """
        if self._nodes is None:
            self._nodes = X
            m = X.shape[0]
            for i in range(m):
                self._add_node_to_tree(X[i, :], i)
            return np.arange(0, self._nodes.shape[0])
        else:
            n = self._nodes.shape[0]
            m = X.shape[0]
            self._nodes = np.concatenate([self._nodes, X], axis=0)
            for i in range(m):
                self._add_node_to_tree(X[i, :], i+n)
            return np.arange(n, self._nodes.shape[0])

    def _add_node_to_tree(self, X: np.ndarray, ix: int):
        """
        Adds a node to the ball tree.

        Parameters:
            X: An array of shape (num_features, )
            ix: The index to attach to the node.
        """
        self._root = self._online_ball_tree_constructor(
            Ball(center=self._projection(X), location_refinement=X),
            ix,
            self._root
        )

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
        n = X.shape[0]
        neighbors = []
        for i in range(n):
            location = X[i, :]
            nn = NearestNeighbors(location, self._root, self._projection)
            neighbors.append(
                [j for _, j in islice(nn, k)]
            )
        return np.array(neighbors)


class NearestNeighbors(Iterator):
    def __init__(self, location: np.ndarray, ball_tree: BallTreeNode, projection: Callable[[np.ndarray], np.ndarray]):
        """
        Given a fixed location_refinement, this class iterates through a Ball Tree,
        returning the points nearest that location_refinement (i.e. returning the points
        in order of proximity to that location_refinement). Assuming the Ball Tree is well
        balanced, each item will be returned efficiently.

        Parameters:
            location: A numpy array representing the query location_refinement. Points in
                the ball tree will be returned in order of proximity to that
                location_refinement.
            ball_tree: A root BallTreeNode whose points are to be queried.
        """
        self._location = location
        self._projection = projection
        self._proj_location = self._projection(location)
        # self._point_heap contains (distance, pt, data) triplets
        self._point_heap = []
        min_distance = self.get_min_distance(ball_tree.ball)
        # self._ball_heaps contains non-empty heaps of
        # (min_distance, BallTreeNode) pairs. The segments spanned by heaps are
        # non-overlapping, and ordered from farthest to nearest.
        self._ball_heaps = [[(min_distance, ball_tree)]]

    def __next__(self) -> Tuple[np.ndarray, Any]:
        """
        Iterates through the points in the ball heap, from nearest to
        farthest.
        """
        if self._point_heap:
            min_pt_dist, pt, data = self._point_heap[0]
            if self._ball_heaps:
                nearest_ball_heap = self._ball_heaps[-1]
                assert nearest_ball_heap, "Ball heaps must be non-empty. %s" % self._ball_heaps
                min_ball_distance, btn = nearest_ball_heap[0]
                if min_pt_dist <= min_ball_distance:
                    hq.heappop(self._point_heap)
                    return pt, data
                else:
                    # we can't guarantee that this pt is closest. Split the ball
                    # heap and try again.
                    self._split_nearest_ball_heap()
                    return self.__next__()
            else:
                hq.heappop(self._point_heap)
                return pt, data
        elif self._ball_heaps:
            # split the ball heaps and try again
            self._split_nearest_ball_heap()
            return self.__next__()
        # we're empty!
        raise StopIteration

    def _split_nearest_ball_heap(self):
        """
        This splits the closest ball tree node on the nearest ball heap
        while maintaining the invariants.
        """
        assert self._ball_heaps, "Unable to process empty ball heap."
        nearest_ball_heap = self._ball_heaps[-1]
        assert nearest_ball_heap, "Ball heaps must be non-empty."
        min_ball_distance, btn = hq.heappop(nearest_ball_heap)
        if btn.is_leaf():
            # Since the closest ball is a leaf, we extract the point and push
            # it onto the point heap.
            distance = self.get_pt_distance(btn.ball)
            hq.heappush(self._point_heap, (distance, btn.ball.location_refinement, btn.data))
            if len(nearest_ball_heap) == 0:
                self._ball_heaps.pop()
        else:
            # Since the closest ball is an internal node (contains sub-balls) we
            # extract each sub-ball and push them individually onto the ball
            # heaps.
            left = btn.left
            left_min_distance, left_max_distance = self.get_min_max_distance(left.ball)
            right = btn.right
            right_min_distance, right_max_distance = self.get_min_max_distance(right.ball)
            if nearest_ball_heap:
                next_nearest_distance, _ = nearest_ball_heap[0]
                if left_max_distance > next_nearest_distance:
                    # left ball overlaps, push it on the heap.
                    hq.heappush(nearest_ball_heap, (left_min_distance, left))
                    next_nearest_distance, _ = nearest_ball_heap[0]
                    if right_max_distance > next_nearest_distance:
                        # right ball also overlaps, push it on the heap.
                        hq.heappush(nearest_ball_heap, (right_min_distance, right))
                        return
                    else:
                        # right ball is disjoint, add it as a new closer heap.
                        self._ball_heaps.append([(right_min_distance, right)])
                        return
                else:
                    if right_max_distance > next_nearest_distance:
                        # right ball overlaps, push it on the heap.
                        hq.heappush(nearest_ball_heap, (right_min_distance, right))
                        next_nearest_distance, _ = nearest_ball_heap[0]
                        if left_max_distance > next_nearest_distance:
                            # left ball now overlaps, push it on the heap.
                            hq.heappush(nearest_ball_heap, (left_min_distance, left))
                            return
                        else:
                            # left ball is disjoint, add it as a new closer heap.
                            self._ball_heaps.append([(left_min_distance, left)])
                            return
                    else:
                        # both left and right balls are disjoint from the rest of
                        # the heap.
                        # Add a new empty ball heap (we need to put the balls
                        # on it).
                        self._ball_heaps.append([])
            # nearest ball heap is empty. Place the left and right balls on it
            if left_max_distance <= right_min_distance:
                # balls are non overlapping.
                self._ball_heaps[-1].append((right_min_distance, right))
                self._ball_heaps.append([(left_min_distance, left)])
            elif right_max_distance <= left_min_distance:
                # balls are non overlapping.
                self._ball_heaps[-1].append((left_min_distance, left))
                self._ball_heaps.append([(right_min_distance, right)])
            else:
                # balls are overlapping:
                hq.heappush(self._ball_heaps[-1], (left_min_distance, left))
                hq.heappush(self._ball_heaps[-1], (right_min_distance, right))

    def get_min_max_distance(self, ball: Ball):
        centered_distance, rnd = self.get_centered_distance(ball)
        return (max(0, centered_distance - ball.radius), rnd), \
               (centered_distance+ball.radius, rnd)

    def get_min_distance(self, ball: Ball):
        return self.get_min_max_distance(ball)[0]

    def get_max_distance(self, ball: Ball):
        return self.get_min_max_distance(ball)[1]

    def get_centered_distance(self, ball: Ball):
        return np.sqrt(((ball.center-self._proj_location)**2).sum()), random.random()

    def get_pt_distance(self, ball: Ball):
        return np.sqrt(((ball.location_refinement - self._location) ** 2).sum()), random.random()

    ####################################################
    # Validation methods, only to be used for testing: #
    ####################################################

    def assert_invariants(self):
        """A validation method, only to be used for testing. It verifies
        that all internal invariants are satisfied."""
        for bh in self._ball_heaps:
            NearestNeighbors._assert_heap(bh)
            assert len(bh) > 0, "Ball heaps must be non-empty."
        min_bh_distances = [bh[0][0] for bh in self._ball_heaps]
        max_bh_distances = [
            max(map(lambda mbtn: self.get_max_distance(mbtn[1].ball), bh))
            for bh in self._ball_heaps
        ]
        for m, M in zip(min_bh_distances, max_bh_distances):
            assert m <= M, "min distance should be smaller than max."
        for i in range(len(self._ball_heaps)-1):
            assert max_bh_distances[i+1] <= min_bh_distances[i], \
                "Ball heaps must not overlap"

        if self._point_heap:
            self._assert_heap(self._point_heap)
            min_pt_distance, *_ = self._point_heap[0]

    @staticmethod
    def _assert_heap(heap):
        """Validates that a list satisfies the heap invariants."""
        for k in range(len(heap)):
            if 2 * k + 1 < len(heap):
                assert heap[k] <= heap[2 * k + 1]
            if 2 * k + 2 < len(heap):
                assert heap[k] <= heap[2 * k + 2]
