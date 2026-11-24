from .greedy import greedy, greedy_assign
from .hungarian import hungarian, hungarian_custom, hungarian_scipy
from .random_assign import manual_assign, manual_pairs, random_assign, random_pairs

BASELINE = "Ngẫu nhiên (Random)"

ALGORITHMS = {
    "Hungarian (SciPy)": hungarian_scipy,
    "Hungarian (Tự cài O(n³))": hungarian_custom,
    "Greedy Heuristic": greedy_assign,
    "Ngẫu nhiên (Random)": random_assign,
    "Thủ công (Tuần tự)": manual_assign,
}

PAIR_ALGORITHMS = {
    "Hungarian (SciPy)": hungarian,
    "Hungarian (Tự cài O(n³))": lambda c: hungarian(c, use_custom=True),
    "Greedy Heuristic": greedy,
    "Ngẫu nhiên (Random)": random_pairs,
    "Thủ công (Tuần tự)": manual_pairs,
}


def run_algorithm(name: str, cost_matrix, seed: int | None = None):
    """Trả về tuple (assignments, total_cost)."""
    if name not in ALGORITHMS:
        raise ValueError(f"Thuật toán không tồn tại: {name}")
    func = ALGORITHMS[name]
    if name == "Ngẫu nhiên (Random)":
        return func(cost_matrix, seed=seed)
    return func(cost_matrix)


def get_assignments(
    name: str, cost_matrix, seed: int | None = None
) -> list[tuple[int, int]]:
    """Trả về trực tiếp list cặp [(driver_idx, order_idx), ...] cho giao diện."""
    assignments, _ = run_algorithm(name, cost_matrix, seed=seed)
    return assignments


__all__ = [
    "hungarian",
    "hungarian_scipy",
    "hungarian_custom",
    "greedy",
    "greedy_assign",
    "random_pairs",
    "random_assign",
    "manual_pairs",
    "manual_assign",
    "ALGORITHMS",
    "PAIR_ALGORITHMS",
    "run_algorithm",
    "get_assignments",
]