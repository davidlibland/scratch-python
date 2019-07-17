import numpy as np

from src.utils.incremental_linkage_graph import Node, IncrementalLinkageGraph


def test_node_storage():
    """Tests that the IncrementalLinkageGraph stores nodes correctly."""
    nodes = [
        Node(
            np.array([[2, 3, 1]]),
            "node a"
        ),
        Node(
            np.array([[4, 1, 2]]),
            "node b"
        ),
        Node(
            np.array([[4, 1, 5]]),
            "node c"
        )
    ]

    linkage_graph = IncrementalLinkageGraph(
        distance_linkage_cost
    )
    for i, n in enumerate(nodes):
        ix, links = linkage_graph.add_node(n)
        assert linkage_graph.get_nodes(list(range(ix+1))) == nodes[:i+1]


def test_correct_linkages_1():
    """Tests that the IncrementalLinkageGraph stores nodes correctly."""
    nodes = [
        Node(
            np.array([[1, 0, 0]]),
            "node a"
        ),
        Node(
            np.array([[0, 1, 0]]),
            "node b"
        ),
        Node(
            np.array([[0, 0, 1]]),
            "node c"
        )
    ]

    linkage_graph = IncrementalLinkageGraph(
        distance_linkage_cost,
        2
    )
    assert linkage_graph.num_nodes == 0, "Wrong number of nodes!"
    ix, links = linkage_graph.add_node(nodes[0])
    assert linkage_graph.num_nodes == 1, "Wrong number of nodes!"
    assert ix == 0 and links == [], "Initial insertion failed."
    ix, links = linkage_graph.add_node(nodes[1])
    assert linkage_graph.num_nodes == 2, "Wrong number of nodes!"
    assert ix == 1 and links == [nodes[0]], "Second insertion failed."
    ix, links = linkage_graph.add_node(nodes[2])
    assert linkage_graph.num_nodes == 3, "Wrong number of nodes!"
    sorted_links = sorted(links, key=lambda node: node.data)
    expected_links = sorted([nodes[0], nodes[1]], key=lambda node: node.data)
    assert ix == 2 and sorted_links == expected_links, \
        "Final insertion failed."

    assert linkage_graph.get_nodes([0, 1, 2]) == nodes, \
        "Nodes stored incorrectly."

    assert [(i, j) for i, j, _ in sorted(linkage_graph.edges)] == \
           [(1, 0), (2, 0), (2, 1)], \
        "Incorrect linkage edges."


def test_correct_linkages_2():
    """Tests that the IncrementalLinkageGraph stores nodes correctly."""
    nodes = [
        Node(
            np.array([[1, 0]]),
            "node a"
        ),
        Node(
            np.array([[0, 1]]),
            "node b"
        ),
        Node(
            np.array([[-1, 0]]),
            "node c"
        ),
        Node(
            np.array([[0, -1]]),
            "node d"
        )
    ]

    linkage_graph = IncrementalLinkageGraph(
        distance_linkage_cost,
        float(np.sqrt(2)) + .1
    )
    ix, links = linkage_graph.add_node(nodes[0])
    assert ix == 0 and links == [], "Initial insertion failed."
    ix, links = linkage_graph.add_node(nodes[1])
    assert ix == 1 and links == [nodes[0]], "Second insertion failed."
    ix, links = linkage_graph.add_node(nodes[2])
    assert ix == 2 and links == [nodes[1]], \
        "Third insertion failed."
    ix, links = linkage_graph.add_node(nodes[3])
    sorted_links = sorted(links, key=lambda node: node.data)
    expected_links = sorted([nodes[0], nodes[2]], key=lambda node: node.data)
    assert ix == 3 and sorted_links == expected_links, \
        "Final insertion failed."

    assert linkage_graph.get_nodes([0, 1, 2, 3]) == nodes, \
        "Nodes stored incorrectly."

    assert [(i, j) for i, j, _ in sorted(linkage_graph.edges)] == \
           [(1, 0), (2, 1), (3, 0), (3, 2)], \
        "Incorrect linkage edges."

    assert (linkage_graph.get_adjacency_matrix().toarray() ==
        np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])).all(), "Incorrect adjacency matrix."

def distance_linkage_cost(left: np.ndarray, right: np.ndarray):
    """Returns the distance."""
    distances = np.sqrt(np.sum((left - right)**2, axis=1))
    return distances
