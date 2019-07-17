import random


class HashTable:
    def __init__(self, num_buckets=2**10, hash_function=None):
        self._table = [[] for _ in range(num_buckets)]
        self._n = num_buckets
        if hash_function is None:
            hash_function = hash
        self._hash_function = hash_function

    def _internal_key(self, key) -> int:
        return self._hash_function(key) % self._n

    def _getitem_internal(self, key, i):
        # Search the current list for the key:
        for j, (k, v) in enumerate(self._table[i]):
            if k == key:
                return j, v
        raise KeyError

    def __setitem__(self, key, value):
        i = self._internal_key(key)
        try:
            # Update existing values:
            j, _ = self._getitem_internal(key, i)
            self._table[i][j] = (key, value)
        except KeyError:
            self._table[i].append((key, value))

    def __getitem__(self, key):
        i = self._internal_key(key)
        _, v = self._getitem_internal(key, i)
        return v

    def __delitem__(self, key):
        i = self._internal_key(key)
        j, _ = self._getitem_internal(key, i)
        left = self._table[i][:j]
        right = self._table[i][j+1:]
        self._table[i] = left + right

    def to_dict(self):
        pairs = []
        for t in self._table:
            pairs.extend(t)
        return dict(pairs)


def test_crud(num_buckets=1, num_ops=25):
    good_container = dict()
    test_container = HashTable(num_buckets=num_buckets)
    keys = "abcd"
    ops = "cdr"
    log = []
    for _ in range(num_ops):
        op = random.choice(ops)
        if op == "c":
            k = random.choice(keys)
            v = random.randint(0, 2**20)
            log.append((op, k, v))
            good_container[k] = v
            test_container[k] = v
        elif op == "r":
            k = random.choice(keys)
            log.append((op, k))
            if k in good_container:
                assert good_container[k] == test_container[k], \
                    "Failed at op %s" % log[-5:]
            else:
                try:
                    x = test_container[k]
                    raise AssertionError("Failed at op %s" % log[-5:])
                except KeyError:
                    pass
        elif op == "d":
            k = random.choice(keys)
            log.append((op, k))
            if k in good_container:
                del good_container[k]
                del test_container[k]
            else:
                try:
                    del test_container[k]
                    raise AssertionError("Failed at op %s" % log[-5:])
                except KeyError:
                    pass
        assert good_container == test_container.to_dict()
