from src.utils.elasticsearch.index_utils import length_weighting
from src.verticals.isaac_semantic.model_params import (
    NAME_LENGTH_SHIFT,
    NAME_LENGTH_SCALE,
    MIN_P_WEIGHT,
    MAX_P_WEIGHT,
    INITIAL_SHINGLE_WEIGHT_FACTOR,
    NORMAL_NAME_WEIGHT_FACTOR,
    PREFERRED_NAME_WEIGHT_FACTOR,
)
lw = length_weighting( 
    NAME_LENGTH_SCALE, 
    NAME_LENGTH_SHIFT, 
    MIN_P_WEIGHT, 
    MAX_P_WEIGHT, 
)
weights = [lw(i) for i in range(500)]


import heapq as hp

class Heap:
    def __init__(self, *elems):
        self._elems = list(elems)
        hp.heapify(self._elems)
        self._dels = []
    
    def pop(self):
        self.clean()
        return hp.heappop(self._elems)
    
    def remove(self, x):
        hp.heappush(self._dels, x)
    
    def clean(self):
        while self._dels and self._elems[0] == self._dels[0]:
            hp.heappop(self._elems)
            hp.heappop(self._dels)
        
    def peek(self):
        self.clean()
        return self._elems[0]
        
    def __len__(self):
        return len(self._elems) - len(self._dels)
    
    def __repr__(self):
        return "%s - %s" % (self._elems, self._dels)
    
    
from collections import deque

class MaxIter:
    def __init__(self, *elems):
        self._elems = elems
        
    def __iter__(self):
        min_heap = Heap(*[-x for x in self._elems])
        for x in self._elems:
            m = -min_heap.peek()
            if x == m:
                min_heap.pop()
            else:
                min_heap.remove(-x)
            yield x, m

import random

def test_max_iter(n=25, a=-3, b = 10):
    elems = [random.randint(a, b) for _ in range(n)]
    print(elems)
    for i, (x, m) in enumerate(iter(MaxIter(*elems))):
        print(elems[i], x, m)
        assert x == elems[i], "Wrong iteration order"
        assert m == max(elems[i:]), "Wrong max"
    assert i + 1 == len(elems), "Wrong length: %d vs %s" % (i, len(elems))