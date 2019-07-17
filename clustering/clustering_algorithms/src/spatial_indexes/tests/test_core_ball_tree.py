from typing import Callable, Optional, Any

import numpy as np

from src.spatial_indexes.core_ball_tree import (
    BallTreeNode, Ball,
    repair_internal_node, insert_as_sibling, NodeTypes,
)


def test_insert_as_sibling(num_dims=2):
    """Tests insertion of siblings"""
    # First we insert at a root:
    right_node = BallTreeNode(
        ball=get_ball()
    )
    left_node = insert_as_sibling(get_ball(), right_node)
    root_node = left_node.parent
    assert root_node.left == left_node
    assert left_node.parent == root_node
    assert root_node.right == right_node
    assert right_node.parent == root_node

    # Insert again at the right node:
    new_left_node = insert_as_sibling(get_ball(), right_node)
    right_parent_node = new_left_node.parent
    assert right_parent_node.left == new_left_node
    assert right_parent_node.right == right_node
    assert root_node.right == right_parent_node
    assert right_parent_node.parent == root_node

    # Insert again at the left node:
    new_left_node = insert_as_sibling(get_ball(), left_node)
    left_parent_node = new_left_node.parent
    assert left_parent_node.left == new_left_node
    assert left_parent_node.right == left_node
    assert root_node.right == right_parent_node
    assert right_parent_node.parent == root_node
    assert root_node.left == left_parent_node
    assert left_parent_node.parent == root_node

    # Verify nested structure:
    validate_bounding_structure(root_node.get_root())


def test_repair_node(num_dims=10, num_samples=100):
    """
    Runs a non-deterministic test to confirm that parent balls surround both
    child balls.
    """
    # First test arbitrary balls:
    left_node = BallTreeNode(
        ball=get_ball(num_dims=num_dims, radius=2)
    )
    right_node = BallTreeNode(
        ball=get_ball(num_dims=num_dims, radius=3)
    )
    parent_node = BallTreeNode(
        ball=None,
        left=left_node,
        right=right_node
    )
    left_node.parent=parent_node
    right_node.parent=parent_node
    repair_internal_node(parent_node)

    # test samples:
    left_samples = sample_pts_from_ball(left_node.ball, num_samples)
    assert_in_ball(left_node.ball, left_samples)
    assert_in_ball(parent_node.ball, left_samples)
    right_samples = sample_pts_from_ball(right_node.ball, num_samples)
    assert_in_ball(right_node.ball, right_samples)
    assert_in_ball(parent_node.ball, right_samples)

    # Next test co-located balls (edge case):
    left_node = BallTreeNode(
        ball=get_ball(num_dims=num_dims, radius=2)
    )
    right_node = BallTreeNode(
        ball=Ball(
            center=left_node.ball.center,
            radius=3
        )
    )
    parent_node = BallTreeNode(
        ball=None,
        left=left_node,
        right=right_node
    )
    left_node.parent=parent_node
    right_node.parent=parent_node
    repair_internal_node(parent_node)

    # test samples:
    left_samples = sample_pts_from_ball(left_node.ball, num_samples)
    assert_in_ball(left_node.ball, left_samples)
    assert_in_ball(parent_node.ball, left_samples)
    right_samples = sample_pts_from_ball(right_node.ball, num_samples)
    assert_in_ball(right_node.ball, right_samples)
    assert_in_ball(parent_node.ball, right_samples)

    # Verify nested structure:
    validate_bounding_structure(parent_node.get_root())



###########
# Helpers #
###########

def get_ball(bound=3, num_dims=10, radius=1e-3):
    return Ball(
        center=np.random.uniform(-bound, bound, size=num_dims),
        radius=radius
    )

def sample_pts_from_ball(ball: Ball, num_samples=100):
    """
    Samples from the ball (non-uniformly in space, and uniformly in radius).
    """
    num_dims = ball.center.size
    unnorm_dirs = np.random.normal(size=[num_samples, num_dims])
    norms = np.sqrt((unnorm_dirs**2).sum(axis=1, keepdims=True))
    dirs = unnorm_dirs/norms
    dists = np.random.uniform(0, ball.radius, size=[num_samples, 1])
    samples = dirs*dists + ball.center.reshape([1, -1])
    return samples


def assert_in_ball(ball: Ball, samples: np.ndarray):
    """
    Asserts that the samples lie within the given ball.
    """
    segments = samples - ball.center.reshape([1, -1])
    distances = np.sqrt((segments**2).sum(axis=1))
    assert (distances <= ball.radius).all(), "Sample lay outside of ball."


def validate_bounding_structure(root):
    """Verifies that balls are appropriately nested."""
    if root.is_leaf():
        return
    assert balls_are_ordered(root.ball, root.left.ball)
    assert balls_are_ordered(root.ball, root.right.ball)
    validate_bounding_structure(root.left)
    validate_bounding_structure(root.right)


def balls_are_ordered(big_ball: Ball, small_ball: Ball) -> bool:
    """Returns True iff the big_ball contains the small_ball."""
    segment = small_ball.center - big_ball.center
    distance = np.sqrt((segment ** 2).sum())
    return big_ball.radius >= distance + small_ball.radius


def generate_tests_for(
        online_ball_tree_constructor:
        Callable[[Ball, Optional[Any], Optional[BallTreeNode]], BallTreeNode],
        max_num_balls=50,
        num_dims=10
) -> Callable[[], None]:
    """
    Generates test suite for an online ball tree constructor (which constructs
    the ball tree incrementally via insertion.
    
    Parameters:
        online_ball_tree_constructor: A function which takes a ball, 
            and possibly a 
        num_balls: The number of balls to test with.

    Returns:
        A function representing a test suite.
    """
    def test(num_balls):
        # Build the tree:
        root = None
        for i in range(num_balls):
            ball = get_ball(num_dims=num_dims)
            root = online_ball_tree_constructor(ball, i, root)

        # Assess the structure of the tree:
        seen_nodes = set()
        def assert_tree(root: BallTreeNode, parent=None):
            assert root.parent is parent, "Child node doesn't remember parent."
            type = root.validate_type()
            if root.parent is None:
                assert NodeTypes.ROOT in type

            if root.left is None:
                assert root.right is None, "Must have zero or two children."
                assert root.is_leaf()
                assert NodeTypes.LEAF in type
            if root.right is None:
                assert root.left is None, "Must have zero or two children."
                assert root.is_leaf()
                assert NodeTypes.LEAF in type

            assert isinstance(root.ball, Ball), "Must have a ball at each node."

            # To be a tree, we must never see a node twice:
            assert id(root) not in seen_nodes, "Can't visit a node twice."
            seen_nodes.add(id(root))

            # Validate the children
            if not root.is_leaf():
                assert NodeTypes.LEFT in root.left.validate_type()
                assert_tree(root.left, root)
                assert NodeTypes.RIGHT in root.right.validate_type()
                assert_tree(root.right, root)
        assert_tree(root)

        # Count the number of nodes:
        def count_balls(root: Optional[BallTreeNode]) -> int:
            if root.is_leaf():
                return 1
            return count_balls(root.left) + count_balls(root.right)

        assert count_balls(root) == num_balls, \
            "Incorrect number of balls. Expected %d, found %d" \
            % (num_balls, count_balls(root))

        # Verify Bounding Ball Structure:
        validate_bounding_structure(root), "Balls need to be nested."

    def test_suite():
        # test edge cases:
        test(1)
        test(2)
        test(3)
        test(4)
        test(5)
        # test max:
        test(max_num_balls)
    return test_suite
