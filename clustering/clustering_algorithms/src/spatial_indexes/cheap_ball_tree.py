from src.spatial_indexes.core_ball_tree import (
    Ball, BallTreeNode,
    get_bounding_ball,
    insert_new_ball_with,
)


def find_best_sibling(new_ball: Ball, root: BallTreeNode) -> BallTreeNode:
    """
    Greedily tries to find the sibling which minimizes the insertion cost
    (defined as the increase in volume of the ball tree).

    Parameters:
        new_ball: The ball to insert
        root: The root of the ball tree to insert into.

    Returns:
        The best sibling for the new ball.
    """
    cur_node = root
    cur_bounding_ball = get_bounding_ball(new_ball, cur_node.ball)
    while not cur_node.is_leaf():
        # We can either insert the new ball as a sibling to the current ball
        # or as a sibling to one of it's two children. Compare these, and
        # Choose the lowest.

        # Cost of insertion as sibling of current node:
        insert_here_cost = cur_bounding_ball.vol() \
                           + cur_node.ball.vol()

        # Cost of insertion into left node:
        insert_left_ball = get_bounding_ball(new_ball, cur_node.left.ball)
        insert_left_cost = insert_left_ball.vol() \
                           + get_bounding_ball(cur_node.right.ball, insert_left_ball).vol()

        # Cost of insertion into right node:
        insert_right_ball = get_bounding_ball(new_ball, cur_node.right.ball)
        insert_right_cost = insert_right_ball.vol() \
                           + get_bounding_ball(cur_node.left.ball, insert_right_ball).vol()

        # Branch on minimal cost:
        if insert_here_cost < min(insert_left_cost, insert_right_cost):
            return cur_node
        if insert_left_cost < insert_right_cost:
            cur_node = cur_node.left
            cur_bounding_ball = insert_left_ball
        else:
            cur_node = cur_node.right
            cur_bounding_ball = insert_right_ball
    return cur_node

build_cheap_ball_tree = insert_new_ball_with(find_best_sibling)
