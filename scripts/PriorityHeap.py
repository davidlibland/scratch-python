import heapq as hp

class Heap:
    # Contains keys with priorities, sorted by highest priority.
    def __init__(self, *elems):
        # elems consist of key-priority pairs
        self._elems = list(self._swap_and_neg(*elems)) # lexicographic order
        hp.heapify(self._elems)
        self._neg_priorities = dict(self._neg_snd(*elems))
        self._dels = []
        
    def add_or_update(self, key, priority):
        try:
            self.remove(key)
        except KeyError:
            pass
        self.push(key, priority)
            
    def push(self, key, priority):
        hp.heappush(self._elems, (-priority, key))
    
    def pop(self):
        self.clean()
        return hp.heappop(self._elems)[1]
    
    def remove(self, key):
        pair = self._neg_priorities[key], key
        hp.heappush(self._dels, pair)
    
    def clean(self):
        while self._dels and self._elems[0] == self._dels[0]:
            hp.heappop(self._elems)
            hp.heappop(self._dels)
        
    def peek(self):
        self.clean()
        return self._elems[0][1]
        
    def __len__(self):
        return len(self._elems) - len(self._dels)
    
    def __repr__(self):
        return "%s - %s" % (self._elems, self._dels)
    
    @staticmethod
    def _neg_snd(*elems):
        # swaps [(x1,y1), ...] -> [(x1,-y1), ...]
        for x, y in elems:
            yield x, -y
    
    @staticmethod
    def _swap(*elems):
        # swaps [(x1,y1), ...] -> [(y1,x1), ...]
        for x, y in elems:
            yield y, x
    
    @staticmethod
    def _swap_and_neg(*elems):
        # swaps [(x1,y1), ...] -> [(-y1,x1), ...]
        yield from Heap._swap(*Heap._neg_snd(*elems))

import random

def random_key(k=5):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz",k=k))
    
def test_heap(n=25, a=-5, b=100):
    elems = [(random_key(), random.randint(a, b)) for _ in range(n)]
    h = Heap(*elems)
    s_elems = sorted(elems, key=lambda x: (-x[1], x[0]))
    # Test priority sort:
    for key,_ in s_elems:
        h_k = h.pop()
        assert key == h_k, "Not popping in the correct order. Expected %s, got %s" % (key, h_k)
    
    # Test pushes
    h = Heap()
    for k, p in elems:
        h.push(k, p)
    
    # Test priority sort:
    for key,_ in s_elems:
        assert key == h.pop(), "Not popping in the correct order from push load"
    
    # Test removes:
    rems = set(map(lambda x: x[0], random.choices(elems, k=n//2)))
    h = Heap(*elems)
    for r in rems:
        h.remove(r)
    
    # Test priority sort:
    for key,_ in filter(lambda k_v: k_v[0] not in rems, s_elems):
        assert key == h.pop(), "Not popping in the correct order"
    
    # Test updates:
    h = Heap(*elems)
    u_elms = [(key, random.randint(a, b)) for key, _ in elems] + \
        [(random_key(), random.randint(a, b)) for _ in range(n)]
    s_u_elms = sorted(u_elms, key=lambda x: (-x[1], x[0]))
    for key, p in u_elms:
        h.add_or_update(key, p)
    
    # Test priority sort:
    for key,_ in s_u_elms:
        assert key == h.pop(), "Not popping in the correct order after updates"
    
    
    
    
    
    
    