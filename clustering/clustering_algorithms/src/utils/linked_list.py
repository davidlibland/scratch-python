# -*- coding: utf-8 -*-
"""
A simple linked list implementation.
ToDo: Move this to HdxBase
"""
# Python modules
from collections.abc import Sized, Iterable
from functools import reduce
from typing import Generic, Optional, List, TypeVar

# 3rd party modules
# - N/A

# HDx modules
# - N/A

# Global Vars
L = TypeVar("L")


class LinkedList(Generic[L], Sized, Iterable):
    """
    A basic implementation of a linked list.
    Note: One advantage a linked list offers over python's native list
    implmentation is the ability for linked lists to share tails.
    """
    def __init__(self, head: L,
                 tail: Optional["LinkedList[L]"] = None):
        self._head = head
        self._tail = tail

    def __len__(self) -> int:
        return reduce(lambda x, _: x + 1, self, 0)

    @property
    def head(self) -> L:
        """Returns the first element in the linked list."""
        return self._head

    @property
    def tail(self) -> Optional["LinkedList[L]"]:
        """Returns all but the first element of the linked list."""
        return self._tail

    @tail.setter
    def tail(self, other: Optional["LinkedList[L]"]):
        """Sets the tail of the linked list."""
        self._tail = other

    @property
    def last(self) -> L:
        """Returns the last element of the linked list."""
        return self.last_link.head

    @property
    def last_link(self) -> "LinkedList[L]":
        if self.tail is None:
            return self
        else:
            return self.tail.last_link

    @last_link.setter
    def last_link(self, other: "LinkedList[L]"):
        if self.tail is None:
            self.tail = other.last_link
        else:
            self.tail.last_link = other.last_link

    @classmethod
    def from_list(cls, labels: List[L]) -> Optional["LinkedList[L]"]:
        """Builds a linked list from a list."""
        if len(labels) == 0:
            return None
        else:
            return LinkedList(labels[0], cls.from_list(labels[1:]))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "%s : %s" % (str(self.head), str(self.tail))

    def tails(self):
        """Iterates through the tails of the list."""
        item = self
        while item is not None:
            yield item
            item = item.tail

    def __iter__(self):
        return map(lambda item: item.head, self.tails())
