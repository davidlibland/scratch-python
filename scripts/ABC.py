from collections import deque
from functools import lru_cache
from itertools import combinations, product


def abc(N, K):
    chars = "ABC"
    min_char = min(chars)
    fifo = deque([("", 0)])
    while len(fifo) > 0:
        s, k = fifo.popleft()
        for c in chars:
            if c != min_char:
                smaller_chars = [c_ for c_ in s if c_ < c]
                new_k = k + len(smaller_chars)
                if new_k == K:
                    return s+c + min_char*(N-len(s)-1)
            else:
                new_k = k
            if len(s) + 1 < N:
                new_s = s+c
                fifo.append((new_s, new_k))
    return ""


def test_string(N, K, s):
    assert len(s) == N, "%s is not of length %d" % (s, N)
    good_char_pairs = [(a, b) for a, b in combinations(s, 2) if a < b]
    assert len(good_char_pairs) == K, \
        "%s does not have %d correctly ordered char pairs:\n%s" \
        % (s, K, good_char_pairs)


def test_abc():
    for N in range(31):
        for K in range(N*(N-1)//2):
            s = abc(N, K)
            print(N, K, s)
            if s:
                test_string(N, K, s)


def run_abc():
    for N in range(30, 31):
        for K in range(N*(N-1)//2):
            s = abc(N, K)

@lru_cache()
def abc_dc(N, K, n_a, n_b, n_c):
    # base_case:
    if N == 0 and K == 0:
        return ""
    elif N == 1 and K == 0:
        if (n_a, n_b, n_c) == (1, 0, 0):
            return "A"
        elif (n_a, n_b, n_c) == (0, 1, 0):
            return "B"
        elif (n_a, n_b, n_c) == (0, 0, 1):
            return "C"
    elif N <= 1:
        return None
    N_l = N // 2
    N_r = N - N_l
    for k_l in range(K+1):
        for k_r in range(K+1 - k_l):
            k_diff = K - k_l - k_r
            # get vals of d's to make up the difference and search:
            [
                (nal, nbl, ncl, nar, nbr, ncr)
                for nal, nbl, ncl in ...
                for nar, nbr, ncr in ...
                if k_diff == nal*(nbr+ncr) + nbl*ncr
            ]


def brute_search(N, K):
    for s_s in product(*(["ABC"]*N)):
        s = "".join(s_s)
        good_char_pairs = [(a, b) for a, b in combinations(s, 2) if a < b]
        if len(good_char_pairs) == K:
            print(s)


def get_num(s):
    n = 0
    for i, c in enumerate(s):
        v = {"A":0, "B":1, "C":2}.get(c)
        n += v*(3**i)
    return n


def test_incremental(N):
    s_s = [abc(N, K) for K in range(N*(N-1)//2 + 1)]
    s_s = [s for s in s_s if s != ""]
    for s1, s2 in combinations(s_s, 2):
        assert get_num(s2) > get_num(s1), "%s <= %s" % (s1, s2)