# -*- coding: utf-8 -*-
"""
A default dict which takes the key as an argument to the default factory.
"""
# Python modules
from collections import defaultdict

# 3rd party modules
# N/A

# HDx modules
# N/A


class ConditionedDefaultDict(defaultdict):
    """Essentially a memoized function. The default factory takes the
    key as an argument. The value of a ConditionedDefaultDict over a
    memoized function is that it can be memoized locally rather than globally.
    It is also extremly light-weight.

    Example:
    >>> fibonaci = ConditionedDefaultDict(
    >>>     lambda x: 1 if x == 0 or x == 1 else fibonaci[x-1] + fibonaci[x-2]
    >>> )
    >>> fibonaci[10]
    89

    """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))
        self[key] = value = self.default_factory(key)
        return value
