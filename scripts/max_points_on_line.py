from typing import List, Tuple
from collections import defaultdict

class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        if len(points) < 1:
            return 0
        if len(set(map(tuple, points))) == 1:
            return len(points)
        lines = defaultdict(set)
        for i, x in enumerate(points):
            for j, y in enumerate(points[:i]):
                if x == y:
                    continue
                line = self.get_line(x, y)
                lines[line].update([i, j])
        if len(lines.values()) == 0:
            return 1
        n = max(map(lambda s: len(s), lines.values()))
        return n


    def get_line(self, a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[float, float, float]:
        assert a != b, "Points must be distinct."
        # a_ = a[0], a[1], 1
        # b_ = b[0], b[1], 1
        c = a[1]-b[1], b[0]-a[0], a[0]*b[1]-a[1]*b[0]
        if c[0] != 0:
            return 1, c[1]/c[0], c[2]/c[0]
        elif c[1] != 0:
            return c[0]/c[1], 1, c[2]/c[1]


def test_solution():
    s = Solution()
    points = [[1,1],[2,2],[3,3]]
    assert s.maxPoints(points) == 3

    points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
    assert s.maxPoints(points) == 4

    points = []
    assert s.maxPoints(points) == 0

    points = [[0,0]]
    assert s.maxPoints(points) == 1

    points = [[0,0],[0,0]]
    assert s.maxPoints(points) == 2