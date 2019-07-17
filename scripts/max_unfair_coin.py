import random

import numpy as np
from scipy.special import softmax


class GuessUnfair:
    def __init__(self, n):
        self.logits = np.array([-np.log(n)] * n)
        self.hts = [(1, 1)]*n
        self.avg_rewards = 0
        self.N = 0

    @property
    def probs(self):
        return softmax(self.logits)

    def play_once(self, callback):
        coin_one_hot = np.random.multinomial(1, self.probs, 1).reshape(-1)
        coin_index = np.argmax(coin_one_hot)
        outcome = callback(coin_index)
        self.learn_from_outcome(coin_one_hot, outcome)

    def learn_from_outcome(self, coin_one_hot, outcome):
        reward = self.get_exp_rew(coin_one_hot, outcome)
        biased_rew = reward - self.avg_rewards
        self.logits += biased_rew*(coin_one_hot-self.probs)
        self.avg_rewards = (self.N * self.avg_rewards + reward)/(self.N + 1)
        self.N += 1

    def get_exp_rew(self, coin_one_hot, outcome):
        coin_index = np.argmax(coin_one_hot)
        a, b = self.hts[int(coin_index)]
        if outcome == 1:
            a += 1
        else:
            b += 1
        self.hts[int(coin_index)] = (a, b)
        samples = np.random.beta(a, b, 300)
        return (samples - 0.5).mean()

    def guess_unfair_ix(self):
        return np.argmax(self.logits)


def test_unfair(n=2, p=1, N=100):
    unfair_oh = np.random.multinomial(1, [1/n]*n, 1)
    unfair_ix = np.argmax(unfair_oh)

    def callback(ix):
        q = 0.5
        if ix == unfair_ix:
            q = p
        return int(np.random.choice([0, 1], 1, p=[1-q, q]))

    G = GuessUnfair(n=n)
    for _ in range(N):
        G.play_once(callback)

    assert G.guess_unfair_ix() == unfair_ix, \
        "Guessed %d instead of %d" % (G.guess_unfair_ix(), unfair_ix)
    print("Correctly guessed %d" % unfair_ix)
