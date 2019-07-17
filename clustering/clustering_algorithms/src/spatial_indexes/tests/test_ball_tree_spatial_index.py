import random
from typing import List, Optional, Any, Tuple

import numpy as np

from src.spatial_indexes.ball_tree_spatial_index import (
    NearestNeighbors,
    BallTreeSpatialIndex,
)
from src.spatial_indexes.core_ball_tree import (
    Ball, BallTreeNode,
    get_bounding_ball,
)
from src.spatial_indexes.tests.test_spatial_index import spatial_index_tester


def test_ball_tree():
    """Run the tests on the BallTreeSpatialIndex"""
    spatial_index_tester(BallTreeSpatialIndex)


def test_nearest_neighbors(num_nodes=4):
    """
    Basic test that the NearestNeighbor class functions correctly.
    """
    centers = [(i, i) for i in range(num_nodes)]
    root = build_simple_binary_tree(centers)

    location = np.array([-1])

    nn = NearestNeighbors(location, root, lambda x: x[:10])
    nn.assert_invariants()
    ixs = []
    for loc, i in nn:
        nn.assert_invariants()
        ixs.append(i)
    assert ixs == sorted(ixs), "Neighbors are not being returned in order"


def build_simple_binary_tree(
        center_data_pairs: List[Tuple[float, Any]],
        shuffled=True
) -> Optional[BallTreeNode]:
    """
    Builds a test ball tree for points on a line.

    Args:
        center_data_pairs: Pairs of centers and data to build the ball tree
            with.
        shuffled: Whether to shuffle the centers, or have an ordered binary
            tree.

    Returns:
        A root BallTreeNode representing the data.
    """
    center_data_pairs = sorted(center_data_pairs)
    if len(center_data_pairs) == 0:
        return None
    if len(center_data_pairs) == 1:
        return BallTreeNode(
            ball=Ball(center=np.array([center_data_pairs[0][0]])),
            data=center_data_pairs[0][1]
        )
    h = len(center_data_pairs) // 2
    left_centers = center_data_pairs[:h]
    right_centers = center_data_pairs[h:]
    trees = [
        build_simple_binary_tree(left_centers),
        build_simple_binary_tree(right_centers)
    ]
    if shuffled:
        random.shuffle(trees)
    left_bt, right_bt = trees
    bounding_ball = get_bounding_ball(left_bt.ball, right_bt.ball)
    ball_tree = BallTreeNode(
        ball=bounding_ball,
        left=left_bt,
        right=right_bt
    )
    left_bt.parent=ball_tree
    right_bt.parent=ball_tree
    return ball_tree


def test_compare_with_sk_learn(n_dims=100, num_seeds=10, num_pts=2000):
    """Checks consistency with sk-learn's implementation."""
    seeds = [np.random.randn(1, n_dims) for _ in range(num_seeds)]

    def get_pt():
        seed = random.choice(seeds)
        return seed + np.random.randn(1, n_dims) / 10

    X_list = [get_pt() for _ in range(num_pts)]
    X = np.vstack(X_list)
    bt_index = BallTreeSpatialIndex(projection_dim=10)
    from src.spatial_indexes.sklearn_ball_tree import SKLearnBallTree
    sk_index = SKLearnBallTree()
    bt_index.add_nodes(X)
    sk_index.add_nodes(X)

    pt = get_pt()
    assert bt_index.query(pt, k=5).flatten().tolist() == sk_index.query(pt, k=5).flatten().tolist()