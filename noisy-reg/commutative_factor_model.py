import math
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn

a = 3

eps = 9
lr = 3e-2

n = 200  # Num iterations
N = 200  # Sample size
As = []
As0 = []

num_factors = 3

# good params:
# num_factors = 4
# eps = 0.5
# lr = 3e-2

# num_factors = 2
# eps = 9
# lr = 3e-2

# num_factors = 3
# eps = 9
# lr = 3e-2

factors = [torch.randn(1, requires_grad=True) for _ in range(num_factors)]

for i in range(N):
    X = np.random.normal(size=n)

    Y = a*X + eps*np.random.normal(size=n)

    with torch.no_grad():
        for A in factors:
            A.normal_()
    As0.append([float(A) for A in factors])


    print("computing %d of %d" % (i+1, N))

    for x, y in zip(X, Y):
        for A in factors:
            if A.grad:
                A.grad.zero_()
        yy = x
        for A in factors:
            yy = A*yy
        loss = (yy-y)**2
        loss.backward()
        with torch.no_grad():
            for A in factors:
                A -= A.grad * lr

        # print(A, A.grad)
    As.append([float(A) for A in factors])

import matplotlib.pyplot as plt

As_prod = []
for a_s in As:
    try:
        a_prod = reduce(mul, a_s)
        if math.isfinite(a_prod):
            As_prod.append(a_prod)
    except Exception as exc:
        print("Logged exception: %s" % exc)

plt.hist(As_prod)
plt.show()
plt.hist([a for a in As_prod if abs(a) < 10])
plt.show()

if num_factors == 2:
    scatter_max = 10
    Ax = [a_s[0] for a_s in As if max(map(abs, a_s)) < scatter_max]
    Ay = [a_s[1] for a_s in As if max(map(abs, a_s)) < scatter_max]
    plt.scatter(Ax, Ay)
    plt.show()
    scatter_max = 10
    Ax = [a_s[0] for a_s in As0 if max(map(abs, a_s)) < scatter_max]
    Ay = [a_s[1] for a_s in As0 if max(map(abs, a_s)) < scatter_max]
    plt.scatter(Ax, Ay)
    plt.show()
elif num_factors > 2:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_max = 10
    Ax = [a_s[0] for a_s in As if max(map(abs, a_s)) < scatter_max]
    Ay = [a_s[1] for a_s in As if max(map(abs, a_s)) < scatter_max]
    Az = [a_s[2] for a_s in As if max(map(abs, a_s)) < scatter_max]
    ax.scatter(Ax, Ay, Az)
    lim = max(map(abs, Ax+Ay+Az))
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    plt.show()



print("mean: %s" % np.mean(As_prod))

# PCA:
pca_max = 10
num_pca_comps = 3
from sklearn.decomposition import PCA
pca = PCA(n_components=num_pca_comps)
PCAs = [a_s for a_s in As if max(map(abs, a_s)) < pca_max]
pca.fit(PCAs)

print("Principal components: %s" % pca.components_)
print("PCA singular values: %s" % pca.singular_values_)
to_view = pca.transform(PCAs)


if num_pca_comps == 2:
    scatter_max = 10
    Ax = [a_s[0] for a_s in to_view if max(map(abs, a_s)) < scatter_max]
    Ay = [a_s[1] for a_s in to_view if max(map(abs, a_s)) < scatter_max]
    plt.scatter(Ax, Ay)
    plt.show()
elif num_pca_comps > 2:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter_max = 10
    Ax = [a_s[0] for a_s in to_view if max(map(abs, a_s)) < scatter_max]
    Ay = [a_s[1] for a_s in to_view if max(map(abs, a_s)) < scatter_max]
    Az = [a_s[2] for a_s in to_view if max(map(abs, a_s)) < scatter_max]
    ax.scatter(Ax, Ay, Az)
    lim = max(map(abs, Ax+Ay+Az))
    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)
    plt.show()