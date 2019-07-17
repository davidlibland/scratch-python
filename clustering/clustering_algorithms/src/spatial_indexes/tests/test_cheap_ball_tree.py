import random

import numpy as np

from src.spatial_indexes.ball_tree_spatial_index import NearestNeighbors
from src.spatial_indexes.cheap_ball_tree import build_cheap_ball_tree
from src.spatial_indexes.core_ball_tree import Ball
from src.spatial_indexes.tests.test_core_ball_tree import (
    generate_tests_for,
    get_ball,
)

test_cheap_ball_tree = generate_tests_for(build_cheap_ball_tree)


def test_nearest_neighbors(num_pts=100, num_dims=3):
    """
    Tests that nearest neighbors are correctly computed.
    """
    location = get_ball(num_dims=num_dims).center
    balls = [get_ball(num_dims=num_dims) for _ in range(num_pts)]

    def distance(ball: Ball):
        return np.sqrt(((location-ball.center)**2).sum())
    sorted_balls = sorted(balls, key=distance)
    labeled_balls = list(enumerate(sorted_balls))
    random.shuffle(labeled_balls)

    # build the tree:
    root = None
    for i, ball in labeled_balls:
        root = build_cheap_ball_tree(ball, i, root)

    nn = NearestNeighbors(location, root, lambda x: x[:10])
    nn.assert_invariants()
    ixs = []
    for loc, i in nn:
        nn.assert_invariants()
        ixs.append(i)
    assert ixs == sorted(ixs), "Neighbors are not being returned in order"
