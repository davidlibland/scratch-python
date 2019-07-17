import numpy as np

from src.random_tree import BinaryDecisionTree, mse_loss


def test_1d_tree():
    # Test at depth of 1:
    X = np.array(
        [[2.],
         [15.],
         [5.],
         [2.],
         [7.],
         [-1.],
         [-1.],
         [1.],
         [11.],
         [7.]]
    )
    y = X/2
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=1, attach_data=True)
    print(d)
    assert d.decision_fn.split_pt == 6.0
    assert d.decision_fn.split_feature == 0
    assert np.isclose(d.compute_mean(np.array([1])), np.array([2/3])), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([1])), np.array([2/3])
        )
    assert d.compute_mean(np.array([10])) == np.array([5.]), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([10])), np.array([5.])
        )

    # Test at depth of 2:
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=2, attach_data=True)
    print(d)
    assert d.decision_fn.split_pt == 6.0
    assert d.left.decision_fn.split_pt == 1.5
    assert d.right.decision_fn.split_pt == 9.0
    assert np.isclose(d.compute_mean(np.array([1])), np.array([-1/6])), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([1])), np.array([-1/6])
        )
    assert d.compute_mean(np.array([10])) == np.array([6.5]), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([10])), np.array([6.5])
        )

    # Test at depth of 25:
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=25, attach_data=True)
    print(d)
    for X, y in zip (X, y):
        assert d.compute_mean(np.array(X)) == np.array(y), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array(X)), np.array(y)
        )


def test_2d_tree():
    # Test at depth of 1:
    X = np.array(
        [[ 2,  4],
         [-3, -1],
         [-3, -2],
         [ 4, -2],
         [-3,  1],
         [-3, -2],
         [ 3,  3],
         [ 0,  4],
         [ 1, -2],
         [ 4, -3]]
    )
    y = (X[:, 0]+0.5*X[:, 1]).reshape(-1, 1)
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=1, attach_data=True)
    print(d)
    assert d.decision_fn.split_pt == -1.5, \
        "split pt: %s" % d.decision_fn.split_pt
    assert d.decision_fn.split_feature == 0, \
        "split feature: %s" % d.decision_fn.split_feature
    assert np.isclose(d.compute_mean(np.array([1, 2])), np.array([2 + 2./3])), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([1, 2])), np.array([2 + 2./3])
        )
    assert d.compute_mean(np.array([3, 4])) == np.array([2 + 2./3]), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([3, 4])), np.array([2 + 2./3])
        )

    # Test at depth of 2:
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=2, attach_data=True)
    print(d)
    assert d.decision_fn.split_pt == -1.5, \
        "split pt: %s" % d.decision_fn.split_pt
    assert d.left.decision_fn.split_pt == 0, \
        "left split ot: %s" % d.left.decision_fn.split_pt
    assert d.right.decision_fn.split_pt == 1.5, \
        "right split pt: %s" % d.right.decision_fn.split_pt
    assert np.isclose(d.compute_mean(np.array([1, 2])), np.array([1.])), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([1, 2])), np.array([1.])
        )
    assert d.compute_mean(np.array([3, 4])) == np.array([3.5]), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array([3, 4])), np.array([3.5])
        )

    import matplotlib.pyplot as plt
    plt.scatter(X[:,0],X[:,1])
    plt.show()
    # Test at depth of 25:
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=25, attach_data=True)
    print(d)
    for X, y in zip (X, y):
        assert d.compute_mean(np.array(X)) == np.array(y), \
        "Incorrect mean computed: %s vs %s" % (
                d.compute_mean(np.array(X)), np.array(y)
        )

def test_prune_order():
    # Test at depth of 1:
    X = np.array(
        [[2.],
         [15.],
         [5.],
         [2.],
         [7.],
         [-1.],
         [-1.],
         [1.],
         [11.],
         [7.]]
    )
    y = X/2
    d = BinaryDecisionTree.fit_unpruned_tree(X, y, mse_loss, depth=50, attach_data=True)
    print([(n.loss_delta_if_collapsed, n.total_loss, n.height) for n in d.prune_order()])