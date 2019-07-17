
def quicksort(u, l=None, h=None):
    if l is None:
        l=0
    if h is None:
        h = len(u) - 1
    i = l
    k = h
    if h-l < 2:
        return

    def swap(i, j):
        x = u[j]
        u[j] = u[i]
        u[i] = x
    while i < k:
        if u[i] > u[i + 1]:
            swap(i, i+1)
            i += 1
        else:
            swap(i+1, k)
            k -= 1
    if i > 0:
        assert max(u[:i]) <= u[i], "%d, %s" % (i, u)
    if i < len(u) - 1:
        assert min(u[i:]) >= u[i], "%d, %s" % (i, u)
    quicksort(u, l, i - 1)
    quicksort(u, i+1, h)
    return u


def merge(l, r):
    i = 0
    j = 0
    m = []
    while i < len(l) and j < len(r):
        if l[i] < r[j]:
            m.append(l[i])
            i+=1
        else:
            m.append(r[j])
            j+=1
    m.extend(l[i:])
    m.extend(r[j:])
    return m


def test_merge():
    l1 = sorted("gfgscj euf")
    l2 = sorted("gvhenie")
    assert merge(l1, l2) == sorted(l1+l2)
    assert merge(l2, l1) == sorted(l1+l2)


def mergesort(u):
    n = len(u)
    if n < 2:
        return u
    l = mergesort(u[:n//2])
    r = mergesort(u[n//2:])
    return merge(l, r)


def test_sorts(alg=mergesort):
    l = list("gfvfnfwe893432ncw")
    assert alg(l) == sorted(l)
    assert alg([]) == []
    assert alg([1]) == [1]
    assert alg([2,1]) == [1,2]