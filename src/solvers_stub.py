import numpy as np
from scipy.optimize import linear_sum_assignment

def hungarian(cost, seed=None):
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))

def greedy(cost, seed=None):
    n = cost.shape[0]
    pairs = sorted((cost[i, j], i, j) for i in range(n) for j in range(n))
    used_i, used_j, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_i and j not in used_j:
            out.append((i, j)); used_i.add(i); used_j.add(j)
    return out

def random_assign(cost, seed=None):
    rng = np.random.default_rng(seed)
    return list(enumerate(rng.permutation(cost.shape[0]).tolist()))

SOLVERS = {"Hungarian": hungarian, "Greedy": greedy, "Random": random_assign}