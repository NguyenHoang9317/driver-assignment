"""
QUY ƯỚC SOLVER

Hàm:    solve(cost, seed=None) -> list[tuple[int, int]]
Input:  cost - np.ndarray (n x n), cost[i, j] = chi phí tài xế i nhận đơn j
        seed - int hoặc None (chỉ thuật toán ngẫu nhiên dùng)
Output: danh sách (i, j); mỗi tài xế và mỗi đơn xuất hiện đúng 1 lần
"""

def check_assignment(cost, assignment):
    n = cost.shape[0]
    assert len(assignment) == n, "Thiếu hoặc thừa cặp"
    assert sorted(i for i, _ in assignment) == list(range(n)), "Tài xế trùng/thiếu"
    assert sorted(j for _, j in assignment) == list(range(n)), "Đơn trùng/thiếu"