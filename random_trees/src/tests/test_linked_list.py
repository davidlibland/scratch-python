from src.linked_list import LinkedList


def test_iteration():
    l = [2, 4, 3, 5]
    ll = LinkedList.from_list(l)
    for i, j in zip(l, ll):
        assert i == j
    assert len(l) == len(ll)


def test_merge():
    l = [4,7,2,8,4,6,9,5,5,7,3,2,8]
    l_sorted = sorted(l)
    l_left = LinkedList.from_list(sorted(l[:5]))
    l_right = LinkedList.from_list(sorted(l[5:]))
    ll = LinkedList.merge(l_left, l_right)
    assert len(ll) == len(l)
    for i, j in zip(ll, l_sorted):
        assert i == j


def test_split():
    l = [4,7,2,8,4,6,9,5,5,7,3,2,8]
    ll = LinkedList.from_list(l)
    print(ll.split())


def test_merge_sort():
    l = [4,7,2,8,4,6,9,5,5,7,3,2,8]
    l_sorted = sorted(l)
    ll = LinkedList.from_list(l).merge_sort()
    assert len(ll) == len(l)
    for i, j in zip(ll, l_sorted):
        assert i == j


def test_keyed_merge_sort():
    l = [4,7,2,8,4,6,9,5,5,7,3,2,8]
    l_rsorted = list(reversed(sorted(l)))
    ll = LinkedList.from_list(l).merge_sort(key=lambda x: -x)
    assert len(ll) == len(l)
    for i, j in zip(ll, l_rsorted):
        assert i == j, "%s vs %s" % (ll, l_rsorted)
