import numpy as np


def random_assign(
    cost_matrix: np.ndarray, seed: int | None = None
) -> tuple[list[tuple[int, int]], float]:
    """Baseline phân bổ ngẫu nhiên."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return [], 0.0

    n_drivers, n_orders = cost.shape
    k = min(n_drivers, n_orders)

    rng = np.random.default_rng(seed)
    driver_ids = rng.permutation(n_drivers)[:k]
    order_ids = rng.permutation(n_orders)[:k]

    assignments = sorted(
        [(int(d), int(o)) for d, o in zip(driver_ids, order_ids)],
        key=lambda x: x[0],
    )
    total_cost = float(sum(cost[r, c] for r, c in assignments))
    return assignments, total_cost


def manual_assign(
    cost_matrix: np.ndarray, pairs: list[tuple[int, int]] | None = None
) -> tuple[list[tuple[int, int]], float]:
    """Baseline phân bổ thủ công (mặc định ghép tuần tự theo thứ tự đến trước: 0-0, 1-1,...)."""
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.size == 0:
        return [], 0.0

    n_drivers, n_orders = cost.shape
    k = min(n_drivers, n_orders)

    if pairs is None:
        assignments = [(i, i) for i in range(k)]
    else:
        assignments = [(int(r), int(c)) for r, c in pairs]

    total_cost = float(sum(cost[r, c] for r, c in assignments))
    return assignments, total_cost


def random_pairs(
    cost_matrix: np.ndarray, seed: int | None = None
) -> list[tuple[int, int]]:
    """Trả về trực tiếp list cặp (driver_idx, order_idx) ngẫu nhiên."""
    assignments, _ = random_assign(cost_matrix, seed=seed)
    return assignments


def manual_pairs(
    cost_matrix: np.ndarray, pairs: list[tuple[int, int]] | None = None
) -> list[tuple[int, int]]:
    """Trả về trực tiếp list cặp (driver_idx, order_idx) thủ công."""
    assignments, _ = manual_assign(cost_matrix, pairs=pairs)
    return assignments