import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_scipy(cost_matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Giải bài toán gán tối ưu bằng scipy.optimize.linear_sum_assignment."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return [], 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    assignments = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
    total_cost = float(cost[row_ind, col_ind].sum())
    return assignments, total_cost


def hungarian_custom(cost_matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Tự cài đặt thuật toán Hungarian (Kuhn-Munkres) độ phức tạp O(n^3)."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return [], 0.0

    transposed = False
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        transposed = True

    n, m = cost.shape  # Luôn đảm bảo n <= m
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(m + 1, dtype=float)
    p = np.zeros(m + 1, dtype=int)
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf, dtype=float)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignments: list[tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            r, c = p[j] - 1, j - 1
            if transposed:
                assignments.append((int(c), int(r)))
            else:
                assignments.append((int(r), int(c)))

    assignments.sort(key=lambda x: x[0])
    orig_cost = np.asarray(cost_matrix, dtype=float)
    total_cost = float(sum(orig_cost[r, c] for r, c in assignments))
    return assignments, total_cost

def hungarian(
    cost_matrix: np.ndarray, use_custom: bool = False
) -> list[tuple[int, int]]:
    """Trả về trực tiếp list cặp (driver_idx, order_idx) cho giao diện."""
    if use_custom:
        assignments, _ = hungarian_custom(cost_matrix)
    else:
        assignments, _ = hungarian_scipy(cost_matrix)
    return assignments