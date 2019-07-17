from src.spatial_indexes.ball_tree_spatial_index import BallTreeSpatialIndex
from src.spatial_indexes.core_ball_tree import BallTreeNode
import numpy as np
import timeit
import random

bt_insert_times = []
bt_query_times = []
tree_depth = []
t_pts = []
n_dims = 200
projection_dim = 10
k=5
bt_index = BallTreeSpatialIndex(projection_dim=projection_dim)


def count_depth(btn: BallTreeNode) -> int:
    if btn.is_leaf():
        return 1
    return max(count_depth(btn.left), count_depth(btn.right)) + 1


seeds = [np.random.randn(1, n_dims) for _ in range(10)]
def get_pt():
    seed = random.choice(seeds)
    return seed + np.random.randn(1, n_dims)/10
    # return np.random.randn(1, n_dims)


for i in range(2000):
    print("timing %d" % i)
    pt = get_pt()
    ti = timeit.timeit(
    'bt_index.add_nodes(pt)',
    globals=globals(), number=1)
    pt = get_pt()
    tq = timeit.timeit(
    'bt_index.query(pt, k)',
    globals=globals(), number=1)
    bt_insert_times.append(ti)
    bt_query_times.append(tq)
    t_pts.append(i)
    tree_depth.append(count_depth(bt_index._root))
    print("depth at %d is %d" % (i, tree_depth[-1]))


import matplotlib.pyplot as plt
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax2.scatter(t_pts, bt_insert_times, color="r", label="insertion times")
ax2.scatter(t_pts, bt_query_times, color="b", label="query times")
ax2.legend()
ax1.plot(t_pts, tree_depth, color="black", label="ball tree depth")
ax1.legend()
plt.title("Performance (in secs) of BallTreeSpatialIndex \n"
          "in %d dims, with %d projection dims, and \n%d nearest neighbors."
          % (n_dims, projection_dim, k))
plt.show()