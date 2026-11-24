"""
QUY ƯỚC SOLVER

Hàm:    solve(cost, seed=None) -> list[tuple[int, int]]
Input:  cost - np.ndarray (n x n), cost[i, j] = chi phí tài xế i nhận đơn j
        seed - int hoặc None (chỉ thuật toán ngẫu nhiên dùng)
Output: danh sách (i, j); mỗi tài xế và mỗi đơn xuất hiện đúng 1 lần
"""

def check_assignment(cost, assignment):
    n, m = cost.shape      
    k = min(n, m)          
    drivers = [i for i, _ in assignment]
    orders = [j for _, j in assignment]

    assert len(assignment) == k, f"Cần {k} cặp, nhận {len(assignment)}"
    assert len(set(drivers)) == k and all(0 <= i < n for i in drivers), "Tài xế trùng hoặc ngoài khoảng 0..n-1"
    assert len(set(orders)) == k and all(0 <= j < m for j in orders), "Đơn trùng hoặc ngoài khoảng 0..m-1"