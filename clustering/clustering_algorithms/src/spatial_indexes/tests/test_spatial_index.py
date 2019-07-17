from typing import Callable

import numpy as np

from src.spatial_indexes.abstract_spatial_index import SpatialIndex
from src.spatial_indexes.sklearn_ball_tree import SKLearnBallTree


def spatial_index_tester(spatial_index_factory: Callable[[], SpatialIndex]):
    """
    Basic test suite for spatial index class.

    Parameters:
        spatial_index_factory: A function which returns a SpatialIndex instance
            to test.
    """
    X_1 = np.array(
        [[0.5488135, 0.71518937, 0.60276338],
         [0.54488318, 0.4236548, 0.64589411],
         [0.43758721, 0.891773, 0.96366276],
         [0.38344152, 0.79172504, 0.52889492],
         [0.56804456, 0.92559664, 0.07103606]]
    )
    X_2 = np.array(
        [[0.0871293, 0.0202184, 0.83261985],
         [0.77815675, 0.87001215, 0.97861834],
         [0.79915856, 0.46147936, 0.78052918],
         [0.11827443, 0.63992102, 0.14335329],
         [0.94466892, 0.52184832, 0.41466194]]
    )
    sp = spatial_index_factory()
    indices = sp.add_nodes(X_1)
    assert np.isclose(sp.nodes[indices, :], X_1).all(), \
        "First set of indices is incorrect."
    indices = sp.add_nodes(X_2)
    assert np.isclose(sp.nodes[indices, :], X_2).all(), \
        "Second set of indices is incorrect."
    assert (sp.query(X_1[:1], k=3) == np.array([[0, 3, 1]])).all(), \
        "Failed to find nearest neighbors"


def test_sklearn_ball_tree():
    """Run the tests on the SKLearnBallTree"""
    spatial_index_tester(SKLearnBallTree)
