
def longest_subsequences(s, max_k):
    max_length, prev_subs = 0, [(0, set()) for _ in range(max_k+1)]
    for c in s:

        subs = [(0, set())]  # base case

        for k, (length, chars) in enumerate(prev_subs[1:]):
            s_length, s_chars = prev_subs[k]
            if c in chars and s_length < length:
                new_length = length + 1
                new_chars_set = chars
            else:
                new_length = s_length + 1
                new_chars_set = s_chars | {c}
            subs.append((new_length, new_chars_set))
        max_length = max(max_length, subs[-1][0])
        prev_subs = subs
    return max_length


class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
        max_length = longest_subsequences(s, k)
        return max_length
