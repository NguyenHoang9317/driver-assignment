import numpy as np


def greedy_assign(cost_matrix: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """Thuật toán Greedy heuristic: luôn ưu tiên chọn cặp (tài xế, đơn) có chi phí nhỏ nhất còn lại."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return [], 0.0

    n_drivers, n_orders = cost.shape
    max_pairs = min(n_drivers, n_orders)

    # Sắp xếp tất cả các cặp (i, j) theo chi phí tăng dần
    flat_indices = np.argsort(cost, axis=None)
    used_drivers: set[int] = set()
    used_orders: set[int] = set()
    assignments: list[tuple[int, int]] = []
    total_cost = 0.0

    for idx in flat_indices:
        r, c = divmod(int(idx), n_orders)
        if r not in used_drivers and c not in used_orders:
            used_drivers.add(r)
            used_orders.add(c)
            assignments.append((r, c))
            total_cost += float(cost[r, c])
            if len(assignments) == max_pairs:
                break

    assignments.sort(key=lambda x: x[0])
    return assignments, total_cost

def greedy(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Trả về trực tiếp list cặp (driver_idx, order_idx) cho giao diện."""
    assignments, _ = greedy_assign(cost_matrix)
    return assignments