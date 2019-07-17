import numpy as np
import torch
import torch.nn as nn

a = 3

eps = .5
lr = 1e-2

n = 500  # num iterations
N = 500  # sample size
As = []

A = torch.zeros(1, requires_grad=True)

for i in range(N):
    X = np.random.normal(size=n)

    Y = a*X + eps*np.random.normal(size=n)

    with torch.no_grad():
        A.fill_(3.)

    print("computing %d of %d" % (i+1, N))

    for x, y in zip(X, Y):
        if A.grad:
            A.grad.zero_()
        loss = (A*x-y)**2
        loss.backward()
        with torch.no_grad():
            A -= A.grad * lr

        # print(A, A.grad)
    As.append(float(A))

import matplotlib.pyplot as plt

plt.hist(As)
plt.show()

print("mean: %s" % np.mean(As))

