from collections import deque, Counter

class LRUDeque(deque):
    # Supports append and popleft; appends erase earlier elements
    def __init__(self, elems):
        super().__init__(elems)
        self._cnts = Counter(elems)
    
    def append(self, e):
        self._cnts[e] += 1
        deque.append(self, e)
    
    def popleft(self):
        x = deque.popleft(self)
        self._cnts[x] -= 1
        while self._cnts[x] > 0:
            x = deque.popleft(self)
            self._cnts[x] -= 1
        return x
    
    def __len__(self):
        # Inefficient, but only used for testing.
        return len([x for x, k in self._cnts.items() if k > 0])

def test_lru_deque():
    elems = "abcdaba"
    lr_q = LRUDeque(elems)
    lr_q.append("d")
    lr_q.append("f")
    expected = "cbadf"
    for e in expected:
        assert e == lr_q.popleft()
    assert len(lr_q) == 0