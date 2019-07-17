from enum import Enum
from typing import Optional, Callable, Any

import numpy as np
from dataclasses import dataclass


DEFAULT_BALL_RADIUS = 1e-3


@dataclass
class Ball:
    center: np.ndarray  # a vector of shape (num_features,)
    location_refinement: Optional[np.ndarray]=None  # a vector of shape (num_features,)
    radius: float=DEFAULT_BALL_RADIUS

    def __post_init__(self):
        assert self.radius >= 0, "Balls must have non-negative radius."
        if self.location_refinement is None:
            self.location_refinement = self.center

    def __eq__(self, other: "Ball"):
        if self.radius != other.radius:
            return False
        if not np.isclose(self.center-other.center, 0).all():
            return False
        return True

    def num_dims(self) -> int:
        """Returns the number of dimensions of the ball."""
        return self.center.size
    
    def vol(self) -> float:
        """Returns the volume of the ball, up to a proportionality constant."""
        return float(np.power(self.radius, self.num_dims()))


class NodeTypes(Enum):
    LEFT="LEFT"
    RIGHT="RIGHT"
    ROOT="ROOT"
    LEAF="LEAF"


@dataclass
class BallTreeNode:
    ball: Ball
    parent: Optional["BallTreeNode"] = None  # None at the root of the tree
    left: Optional["BallTreeNode"] = None  # None at leaves
    right: Optional["BallTreeNode"] = None  # None at leaves
    data: Optional[Any]=None

    def is_left_child(self):
        if self.parent is not None:
            return self.parent.left is self

    def is_right_child(self):
        if self.parent is not None:
            return self.parent.right is self

    def is_leaf(self):
        if self.left is None and self.right is None:
            return True
        return False

    def get_root(self) -> "BallTreeNode":
        root_candidate = self
        while root_candidate.parent is not None:
            root_candidate = root_candidate.parent
        return root_candidate

    def validate_type(self):
        types = set()
        if self.left is None and self.right is None:
            types.add(NodeTypes.LEAF)
        else:
            assert self.left.parent is self, "Left child is orphaned."
            assert self.right.parent is self, "Right child is orphaned."
        if self.parent is None:
            types.add(NodeTypes.ROOT)
        else:
            if self.parent.left is self:
                types.add(NodeTypes.LEFT)
            elif self.parent.right is self:
                types.add(NodeTypes.RIGHT)
            else:
                raise AssertionError("Orphaned Node.")
        return types

    def to_str(self, num_indents=0):
        if self.is_leaf():
            return "%sleaf ball: %s\n%sdata: %s" % \
                   ("\t"*num_indents, self.ball, "\t"*num_indents, self.data)
        return "%sinternal ball: %s\n%sdata: %s\n%sleft:\n%s\n%sright:\n%s" % \
               ("\t"*num_indents, self.ball,
                "\t"*num_indents, self.data,
                "\t"*num_indents, self.left.to_str(num_indents+1),
                "\t"*num_indents, self.right.to_str(num_indents+1))

    def __repr__(self):
        return self.to_str()


def insert_new_ball_with(
        find_best_sibling: Callable[[Ball, BallTreeNode], BallTreeNode]
) -> Callable[[Ball, Optional[Any], Optional[BallTreeNode]], BallTreeNode]:
    """
    Given an algorithm to find the best sibling at which to insert the ball,
    this function performs the actual insertion.
    """
    def helper(new_ball: Ball, data: Optional[Any]=None, root: Optional[BallTreeNode]=None):
        if root is None:
            return BallTreeNode(
                ball=new_ball,
                data=data
            )
        sibling = find_best_sibling(new_ball, root)
        insert_as_sibling(new_ball, sibling, data)
        return root.get_root()
    return helper


def repair_parents(node: BallTreeNode):
    """
    Updates the ball of the given node as well as all it's parents,
    assuming both children have valid balls.

    Notes:
        Mutates the node as well as all it's parents

    Parameters:
        node: The BallTreeNode to update.

    Returns:
        None
    """
    while node is not None:
        repair_internal_node(node)
        node = node.parent


def repair_internal_node(node: BallTreeNode):
    """
    Updates the ball of the internal node, assuming both children have
    valid balls.

    Notes:
        Mutates the internal node.
        Since this changes the ball at the given node, all it's parents need to
        be repaired. Try using `repair_parents` instead.


    Parameters:
        node: The BallTreeNode to update.

    Returns:
        None
    """
    assert not node.is_leaf(), \
        "Can't run `repair_internal_node` on leaf node."
    node.ball = get_bounding_ball(node.left.ball, node.right.ball)
    

def get_bounding_ball(left_ball: Ball, right_ball: Ball, tolerance=1e-10) -> Ball:
    """Returns a new ball containing both the left and right balls."""
    l_center = left_ball.center
    r_center = right_ball.center
    if np.isclose(l_center-r_center, 0).all():
        # Both balls are centered at the same point. Just use the large radius.
        # Add the following tolerance (guaranteed to be small) to ensure that
        # the new ball still bounds both children.
        error = np.sqrt(((l_center-r_center)**2).sum())
        if left_ball.radius > right_ball.radius:
            new_center = left_ball.center
            new_radius = left_ball.radius + error
        else:
            new_center = right_ball.center
            new_radius = right_ball.radius + error
    else:
        segment = r_center-l_center
        distance = np.sqrt((segment**2).sum())
        renormed_segment = segment/distance
        if distance + left_ball.radius <= right_ball.radius:
            # right ball contains left ball:
            new_center = right_ball.center
            new_radius = right_ball.radius
        elif distance + right_ball.radius <= left_ball.radius:
            # left ball contains right ball:
            new_center = left_ball.center
            new_radius = left_ball.radius
        else:
            furthest_left = l_center - renormed_segment*left_ball.radius
            furthest_right = r_center + renormed_segment*right_ball.radius
            new_center = 0.5*(furthest_left+furthest_right)
            new_radius = 0.5*(distance+left_ball.radius+right_ball.radius)+tolerance
    new_ball = Ball(center=new_center, radius=new_radius)
    return new_ball


def insert_as_sibling(
        new_ball: Ball,
        sibling: BallTreeNode,
        data: Optional[Any]=None
) -> BallTreeNode:
    """
    Inserts a new ball as the left sibling to the given node. Mutates the
    existing tree.

    Parameters:
        new_ball: The new ball to insert
        sibling: The sibling.
        data: Optional data to attach at the node.

    Returns:
        None
    """
    # new_ball will be added as a leaf node.
    new_node = BallTreeNode(ball=new_ball, data=data)
    # Make a new parent node:
    if sibling.parent is None:
        # We are at the root of the tree
        new_parent = BallTreeNode(
            ball=None,  # this violates the type constraints:
            # we will need to repair it.
            parent=None,  # this is the new root.
            left=new_node,
            right=sibling
        )
        new_parent.left.parent = new_parent
        new_parent.right.parent = new_parent
        # assert NodeTypes.ROOT in new_parent.validate_type()
    elif sibling.is_left_child():
        # We are at an internal left-child node,
        new_parent = BallTreeNode(
            ball=None,  # this violates the type constraints:
            # we will need to repair it.
            parent=sibling.parent,  # this is the new root.
            left=new_node,
            right=sibling
        )
        new_parent.left.parent = new_parent
        new_parent.right.parent = new_parent
        new_parent.parent.left = new_parent
        # assert NodeTypes.LEFT in new_parent.validate_type()
    elif sibling.is_right_child():
        # We are at an internal right-child node,
        new_parent = BallTreeNode(
            ball=None,  # this violates the type constraints:
            # we will need to repair it.
            parent=sibling.parent,  # this is the new root.
            right=sibling,
            left=new_node
        )
        new_parent.left.parent = new_parent
        new_parent.right.parent = new_parent
        new_parent.parent.right = new_parent
        # assert NodeTypes.RIGHT in new_parent.validate_type()
    else:
        raise ValueError("Node %s is neither a root node, "
                         "nor a left of right child" % sibling)
    # assert NodeTypes.LEAF in new_node.validate_type()
    # assert NodeTypes.RIGHT in sibling.validate_type()
    # assert NodeTypes.LEFT in new_node.validate_type()
    repair_parents(new_parent)
    return new_node
