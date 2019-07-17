from typing import List
from collections import defaultdict


class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        windows = defaultdict(lambda: 0)
        for width in range(1, len(nums)+1):
            for start in range(len(nums)-width+1):
                end = start+width
                for i in range(start, end):
                    left = 1 if start==0 else nums[start-1]
                    right = 1 if end==len(nums) else nums[end]
                    windows[(start, end)] = max(
                        windows[(start, end)],
                        windows[(start, i)] + windows[(i+1, end)]
                        + nums[i]*left*right
                    )
        return windows[(0, len(nums))]