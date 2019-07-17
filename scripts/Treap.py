class Binner:
    def insert(self, n):
        raise NotImplementedError()

    def get_bin(self, n):
        raise NotImplementedError()

class SimpleBinner(Binner):
    def __init__(self):
        self._vals = []

    def insert(self, n):
        self._vals.append(n)

    def get_bin(self, n):
        try:
            left = max(filter(lambda x: x < n, self._vals))
        except:
            left = None
        try:
            right = min(filter(lambda x: x > n, self._vals))
        except:
            right = None
        return left, right

class Treap:
    def __init__(self, n=None, priority: float=None, left: "Treap"=None, right: "Treap"=None):
        priority = random.random() if priority is None else priority
        self.n = n
        self.priority = priority
        self.left = left
        self.right = right
    
    def insert(self, n, priority=None):
        priority = random.random() if priority is None else priority
        if self is None or self.n is None:
            return Treap(n, priority)
        if n >= self.n:
            left = self.left
            right = Treap.insert(self.right, n, priority)
            return Treap.balance_right(self.n, self.priority, left, right)
        else:
            left = Treap.insert(self.left, n, priority)
            right = self.right
            return Treap.balance_left(self.n, self.priority, left, right)

    @staticmethod
    def balance_left(n, priority, left, right):
        if left is not None and left.priority > priority:
            n, priority, left, right = left.n, left.priority, left.left, Treap(n, priority, left.right, right)
        return Treap(n, priority, left, right)

    @staticmethod
    def balance_right(n, priority, left, right):
        if right is not None and right.priority > priority:
            n, priority, left, right = right.n, right.priority, Treap(n, priority, left, right.left), right.right
        return Treap(n, priority, left, right)
    
    def __iter__(self):
        if self.left is not None:
            yield from iter(self.left)
        if self.n is not None:
            yield self.n
        if self.right is not None:
            yield from iter(self.right)
    
    def get_bin(self, n, left=None, right=None):
        if self.n is None:
            return None, None
        # left is to the left of everything in self
        # right is to the right of everything in self
        assert n != self.n, "%s is in the treap" % n
        if n < self.n:
            if self.left is None:
                return left, self.n
            return self.left.get_bin(n, left, self.n)
        else:
            if self.right is None:
                return self.n, right
            return self.right.get_bin(n, self.n, right)
    
    def __repr__(self):
        return repr(self.n)
    
    def display(self, p=False):
        l_disp = "•" if self.left is None else self.left.display(p)
        r_disp = "•" if self.right is None else self.right.display(p)
        if p:
            return "(%s<- %s, %.3f -> %s)" % (l_disp, self.n, self.priority, r_disp)
        return "(%s<- %s -> %s)" % (l_disp, self.n, r_disp)

def test_treap(Treap, n=25, disp=False, p=True):
    l = [(random.random(), i) for i in range(n)]
    l = sorted(l)
    l = [i for _,i in l]
    t = Treap()
    for i in l:
        t = t.insert(i)
    if disp:
        print(t.display(p))
    assert list(t) == sorted(l)
    
    def test_heap(t):
        if t.left is not None:
            assert t.priority >= t.left.priority, "Failed left heap"
            test_heap(t.left)
        if t.right is not None:
            assert t.priority >= t.right.priority, "Failed right heap"
            test_heap(t.right)
    
    test_heap(t)
        