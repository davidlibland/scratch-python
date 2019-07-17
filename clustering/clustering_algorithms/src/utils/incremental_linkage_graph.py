from typing import List, Any, Callable, Tuple, Optional

import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix

from src.spatial_indexes.abstract_spatial_index import SpatialIndex
from src.spatial_indexes.ball_tree_spatial_index import BallTreeSpatialIndex
from src.spatial_indexes.sklearn_ball_tree import SKLearnBallTree


@dataclass
class Node:
    location: np.ndarray  # an array of shape (1, num_features)
    data: Any

    def validation_errors(self) -> Optional[str]:
        if len(self.location.shape) != 2 or self.location.shape[0] != 1:
            return "The location_refinement should be of shape (1, num_features), not %s" \
                % self.location.shape
        return None

    def __post_init__(self):
        error = self.validation_errors()
        if error:
            raise ValueError(error)

    def __eq__(self, other: "Node"):
        if not np.isclose(self.location, other.location).all():
            return False
        if self.data != other.data:
            return False
        return True


class IncrementalLinkageGraph:
    def __init__(self,
                 link_cost: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 cost_threshold: float=0.5,
                 num_neighbors: int=5):
        """
        Builds a linkage graph incrementally. p

        Parameters:
            link_cost: A function which indicates whether nodes should
                be linked. It takes two arrays `left`, `right` both of shape
                (n, num_features), and returns a cost array of shape (n,)
                whose i^th entry indicates the cost of linking left[i,:]
                with right[i,:].
            cost_threshold: The threshold below which links will be added to the
                graph.
            num_neighbors: A floating point distance, beyond which no
                links will be considered.
        """
        self._link_cost_f = link_cost
        self._cost_threshold = cost_threshold
        # self._spatial_index: SpatialIndex = SKLearnBallTree(num_neighbors)
        self._spatial_index: SpatialIndex = BallTreeSpatialIndex()
        self._node_locations = None
        self._data = []
        self._num_neighbors = num_neighbors
        self._edges = []

    def add_node(self, node: Node) -> Tuple[int, List[Node]]:
        """
        Adds a node to the graph, returns a node index along with a list of
        newly added links.
        """
        X = node.location.reshape([1, -1])
        neighbors = None

        # Compute the neighbors (if relevant)
        if self._node_locations is not None:
            k = min(self._num_neighbors, len(self._data))
            neighbors = self._spatial_index.query(X, k)

        # Add the node to the spatial index
        ix = int(self._spatial_index.add_nodes(X))

        # Update the node locations and data.
        if isinstance(self._node_locations, np.ndarray):
            assert ix == self._node_locations.shape[0] == len(self._data), \
                "Internal Error: the spatial index returned an invalid index."
            self._node_locations = np.concatenate(
                [self._node_locations, X],
                axis=0
            )
        else:
            self._node_locations = X
        self._data.append(node.data)

        if neighbors is not None:
            left_features = np.tile(X, [neighbors.size, 1])
            right_features = self._node_locations[neighbors.flatten(), :]
            all_link_costs = self._link_cost_f(left_features, right_features)
            are_linked = all_link_costs < self._cost_threshold
            link_indices = neighbors.flatten()[are_linked.flatten()]
            link_costs = all_link_costs.flatten()[are_linked.flatten()]

            # Store the edges:
            for jx, cost in zip(link_indices, link_costs):
                self._edges.append((int(ix), int(jx), float(cost)))
            return ix, self.get_nodes(link_indices)
        return ix, []

    def get_nodes(self, indices: List[int]) -> List[Node]:
        results = []
        for i in indices:
            next_node = Node(
                location=self._node_locations[i,:].reshape([1, -1]),
                data=self._data[i]
            )
            results.append(next_node)
        return results

    @property
    def edges(self) -> List[Tuple[int, int, float]]:
        """
        A list of edges, where each edge is denoted by a pair of indices,
        followed by the link cost.

        Returns:
            A list of 3-tuples (i, j, cost_ij) where i and j are the node
            indices and cost_ij is the link cost.
        """
        return self._edges

    @property
    def num_nodes(self):
        return len(self._data)

    def get_adjacency_matrix(self):
        """
        Returns a sparse csr adjacency matrix.
        """
        edge_data = []
        row_ind = []
        col_ind = []
        for i, j, _ in self.edges:
            edge_data.append(1)
            row_ind.append(i)
            col_ind.append(j)
            # add the symmetric edge.
            edge_data.append(1)
            row_ind.append(j)
            col_ind.append(i)
        n = self.num_nodes
        return csr_matrix((edge_data, (row_ind, col_ind)), shape=[n, n])

    def get_cost_matrix(self):
        """
        Returns a sparse csr cost matrix.
        """
        edge_data = []
        row_ind = []
        col_ind = []
        for i, j, cost in self.edges:
            edge_data.append(cost)
            row_ind.append(i)
            col_ind.append(j)
            # add the symmetric edge.
            edge_data.append(cost)
            row_ind.append(j)
            col_ind.append(i)
        n = self.num_nodes
        return csr_matrix((edge_data, (row_ind, col_ind)), shape=[n, n])

de