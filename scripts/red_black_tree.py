from enum import Enum


class RBTree:
    class Color(Enum):
        RED = "R"
        BLACK = "B"

    @property
    def color(self) -> "RBTree.Color":
        raise NotImplementedError

    def insert(self, val) -> "RBTree":
        raise NotImplementedError

    def _insert(self, val) -> "RBTree":
        raise NotImplementedError

    @property
    def is_red(self):
        if self.color == RBTree.Color.RED:
            assert isinstance(self, InternalNode), "Leaf nodes must be black"
            return True
        return False

    def __iter__(self):
        yield from []

    def _count_black(self):
        raise NotImplementedError


class NilNode(RBTree):
    @property
    def color(self) -> RBTree.Color:
        return RBTree.Color.BLACK

    def insert(self, val) -> RBTree:
        return InternalNode(RBTree.Color.BLACK, val, NilNode(), NilNode())

    def _insert(self, val) -> RBTree:
        return InternalNode(RBTree.Color.RED, val, NilNode(), NilNode())

    def _count_black(self):
        return 1

    def __repr__(self):
        return ""


class InternalNode(RBTree):
    def __init__(self, color: RBTree.Color, val, left: RBTree, right: RBTree):
        self._color = color
        self._val = val
        self._left = left
        self._right = right

    @property
    def color(self) -> RBTree.Color:
        return self._color

    def _insert(self, val) -> "InternalNode":
        if val < self._val:
            left = self._left._insert(val)
            right = self._right
        else:
            left = self._left
            right = self._right._insert(val)
        return InternalNode(self.color, self._val, left, right)._balance()

    def insert(self, val) -> "InternalNode":
        result = self._insert(val)
        result._color = RBTree.Color.BLACK
        result._count_black()
        return result

    def _balance(self) -> "InternalNode":
        if self._left.is_red:
            if self._left._left.is_red:
                Z = self
                Y = self._left
                X = self._left._left
                a = X._left
                b = X._right
                c = Y._right
                d = Z._right
                return InternalNode(
                    RBTree.Color.RED,
                    Y._val,
                    InternalNode(RBTree.Color.BLACK, X._val, a, b),
                    InternalNode(RBTree.Color.BLACK, Z._val, c, d)
                )
            elif self._left._right.is_red:
                Z = self
                X = self._left
                Y = self._left._right
                a = X._left
                b = Y._left
                c = Y._right
                d = Z._right
                return InternalNode(
                    RBTree.Color.RED,
                    Y._val,
                    InternalNode(RBTree.Color.BLACK, X._val, a, b),
                    InternalNode(RBTree.Color.BLACK, Z._val, c, d)
                )
        if self._right.is_red:
            if self._right._left.is_red:
                X = self
                Z = self._right
                Y = self._right._left
                a = X._left
                b = Y._left
                c = Y._right
                d = Z._right
                return InternalNode(
                    RBTree.Color.RED,
                    Y._val,
                    InternalNode(RBTree.Color.BLACK, X._val, a, b),
                    InternalNode(RBTree.Color.BLACK, Z._val, c, d)
                )
            elif self._right._right.is_red:
                X = self
                Y = self._right
                Z = self._right._right
                a = X._left
                b = Y._left
                c = Z._left
                d = Z._right
                return InternalNode(
                    RBTree.Color.RED,
                    Y._val,
                    InternalNode(RBTree.Color.BLACK, X._val, a, b),
                    InternalNode(RBTree.Color.BLACK, Z._val, c, d)
                )
        return self

    def __iter__(self):
        yield from self._left
        yield self._val
        yield from self._right

    def __repr__(self):
        return "(%s %s %s %s)" % (self._left, self._val, self.color.value, self._right)

    def _count_black(self):
        l_b = self._left._count_black()
        r_b = self._right._count_black()
        assert l_b == r_b, "Balance isn't maintained: %s" % self
        if self.is_red:
            assert not self._left.is_red, "Left is red %s" % self
            assert not self._right.is_red, "Right is red %s" % self
            return l_b
        return l_b + 1


def test_rb_tree():
    from functools import reduce
    empty = NilNode()
    l = [1,6,2,7,3,5,9,5]
    t = reduce(lambda t, x: t.insert(x), l, empty)
    assert sorted(l) == list(t)
    t._count_black()