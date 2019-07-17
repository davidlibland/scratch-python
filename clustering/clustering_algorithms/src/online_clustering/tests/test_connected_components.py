import numpy as np

from src.online_clustering.connected_components import \
    FastConnectedComponentClusters


def distance_linkage_score(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Returns the distance."""
    distances = np.sum((left - right) ** 2, axis=1)
    return np.exp(-distances)


def test_adding_nodes_incrementally():
    """
    Tests the FastConnectedComponentClusters class.
    Tests adding nodes incrementally results in correct clusters.
    """
    cc = FastConnectedComponentClusters(
        match_scores=distance_linkage_score,
        score_threshold=0.5
    )
    nodes = [
        np.array([[1]]),
        np.array([[3]]),
        np.array([[1.1]]),
        np.array([[2.9]]),
    ]
    for i, locs in enumerate(nodes):
        cc.add_nodes(locs)
        assert cc.num_nodes == i+1, "Nodes are not counted correctly."
    labels = list(cc.get_labels())
    a, b, *_ = labels  # grab the first two labels
    assert labels == [a, b, a, b], "Clustering returned incorrect labels."


def test_adding_node_batch():
    """
    Tests the FastConnectedComponentClusters class. Ensures that clustering
    is correct.
    """
    cc = FastConnectedComponentClusters(
        match_scores=distance_linkage_score,
        score_threshold=0.5
    )
    nodes = np.array([[1],[3], [1.1], [2.9]])
    cc.add_nodes(nodes)
    assert cc.num_nodes == 4, "Nodes are not counted correctly."
    labels = list(cc.get_labels())
    a, b, *_ = labels  # grab the first two labels
    assert labels == [a, b, a, b], "Clustering returned incorrect labels."
