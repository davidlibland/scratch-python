# -*- coding: utf-8 -*-
"""
A simple union-find implementation.
ToDo: Move this to HdxBase
ToDo: This can be made more performant by restructuring the forest whenever
    edges are added (whenever we traverse a path, point everything along
    that path to the root).
"""
# Python modules
from collections import defaultdict
from typing import Generic, TypeVar, Mapping, List, MutableMapping

# 3rd party modules
# N/A

# HDx modules
from src.utils.conditioned_defaultdict import ConditionedDefaultDict
from src.utils.linked_list import LinkedList

# Global variables
N = TypeVar("N")
L = TypeVar("L")


class Partition(Generic[N, L], MutableMapping[N, L]):
    """
    A basic union-find data structure. This implements a dict-like interface,
    but the values should be interpreted as partitioning the keys; in more
    detail: the value assigned to a key indicates which component that key is
    assigned to in the partition.

    Unlike a usual dict, note that common values are efficiently shared amongst
    keys, and mutating the value assigned to a given key efficiently updates
    all other keys with the same value. For example:
    >>> partition = Partition({"a": 1, "b": 2, "c": 2, "d": 3})
    >>> partition["a"] = 2
    partition = Partition({"a": 2, "b": 2, "c": 2, "d": 3})
    >>> partition["a"] = 3
    partition = Partition({"a": 3, "b": 3, "c": 3, "d": 3})

    Besides the dict-like interface, two additional methods are provided:

        add_edge: Given two key values, this merges the corresponding
            components. One of the two component labels is chosen arbitrarily
            and the other is relabeled.
             For example, calling add_edge("a", "b") on a
            `Partition` of the form
                Partition({"a": 1, "b": 2, "c": 2, "d": 3})
            could result in
                Partition({"a": 2, "b": 2, "c": 2, "d": 3})
            or
                Partition({"a": 1, "b": 1, "c": 1, "d": 3})

        components: Returns a list of list of keys, describing the partition of
            the keys according to their values. In the above example, the
            components of
                Partition({"a": 1, "b": 2, "c": 2, "d": 3})
            would be:
                [["a"], ["b", "c"], ["d"]],
            and following a call to add_edge("a", "b"), they would be:
                [["a", "b", "c"], ["d"]],
    """
    def __init__(self, initial_components: Mapping[N, L]):
        self._links = ConditionedDefaultDict(
            lambda label: LinkedList(label)
        )
        self._component_forest = dict()
        for node, component in initial_components.items():
            self[node] = component

    def __getitem__(self, node: N):
        return self._component_forest[node].last

    def __setitem__(self, node: N, component: L):
        if node in self._component_forest:
            if self[node] != component:
                old_root = self._component_forest[node].last_link
                new_root = self._links[component]
                # Ensure the new link is cleaved (to become the new root).
                new_root.tail = None
                join_first_to_second(old_root, new_root)
        else:
            self._component_forest[node] = self._links[component]

    def __len__(self):
        return len(self._component_forest)

    def __iter__(self):
        return iter(self._component_forest)

    def __delitem__(self, node: N):
        del(self._component_forest[node])

    def add_edge(self, node1: N, node2: N):
        """
        Merges the components containing node1 and node2. Mutates the data.

        Parameters:
            node1: The node identifying the first component to be merged
            node2: The node identifying the second component to be merged

        Raises:
            ValueError: If one of the two nodes is missing from the data
            structure.
        """
        if node1 not in self._component_forest:
            raise ValueError("Node %s is missing." % node1)
        if node2 not in self._component_forest:
            raise ValueError("Node %s is missing." % node1)
        component1_leaf = self._component_forest[node1]
        component2_leaf = self._component_forest[node2]
        if component1_leaf.last != component2_leaf.last:
            join_first_to_second(
                *sorted([component1_leaf, component2_leaf], key = len)
            )

    def components(self) -> List[List[N]]:
        """
        Return a list of lists of components reflecting the partition of the
        keys according to their values.

        Returns:
            (:class:`list`) each element of which is a list of keys in a given
            component.
        """
        _components = defaultdict(set)
        for key, link in self._component_forest.items():
            _components[link.last].add(key)
        return list(map(list, _components.values()))

###########
# Helpers #
###########


def join_first_to_second(first: LinkedList[L], second: LinkedList[L]):
    """
    Given two linked lists, this method appends the last link of the second
    list to the end of the first list.
    This mutates the first list.

    Parameters:
        first (:class:`LinkedList`): One of the two lists to be joined.
        second (:class:`LinkedList`): One of the two lists to be joined.
    """
    first.last_link = second.last_link
