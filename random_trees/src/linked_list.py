from typing import Iterable, Optional, List, Tuple, Generic, TypeVar

T = TypeVar("T")

class _Nil(Generic[T]):
    def __iter__(self):
        yield from []

    def __len__(self):
        return 0

    def __repr__(self):
        return "Nil"

Nil = _Nil()  # Singleton Nil

class LinkedList(Iterable, Generic[T]):
    def __init__(self, head, tail: Optional["LinkedList"]=None):
        self.head = head
        if tail is None:
            tail = Nil
        self.tail = tail

    def __iter__(self):
        yield self.head
        yield from self.tail

    def __repr__(self):
        return " -> ".join([str(i) for i in self])

    @staticmethod
    def merge(l1: "LinkedList", l2: "LinkedList", key=None) -> "LinkedList":
        if l1 == Nil:
            return l2
        if l2 == Nil:
            return l1
        if (l1.head if key is None else key(l1.head)) < \
                (l2.head if key is None else key(l2.head)):
            return LinkedList(l1.head, LinkedList.merge(l1.tail, l2, key))
        else:
            return LinkedList(l2.head, LinkedList.merge(l1, l2.tail, key))

    def split(self, a: "LinkedList"=Nil, b: "LinkedList"=Nil
              ) -> Tuple["LinkedList", "LinkedList"]:
        aa = LinkedList(self.head, b)
        bb = a
        if self.tail == Nil:
            return aa, bb
        else:
            return self.tail.split(aa, bb)

    def merge_sort(self, key=None) -> "LinkedList":
        if self.tail == Nil:
            return self
        left, right = self.split()
        if left != Nil:
            left = left.merge_sort(key)
        if right != Nil:
            right = right.merge_sort(key)
        return LinkedList.merge(left, right, key)

    @staticmethod
    def from_list(i: List):
        if len(i) == 0:
            return Nil
        return LinkedList(i[0], LinkedList.from_list(i[1:]))

    def __len__(self):
        return 1 + len(self.tail)

    def reverse(self) -> "LinkedList":
        ll = Nil
        for i in self:
            ll = LinkedList(i, ll)
        return ll
