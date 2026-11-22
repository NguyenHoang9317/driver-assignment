"""
Sinh dữ liệu giả lập cho bài toán phân bổ đơn hàng cho tài xế.
"""

import numpy as np
import pandas as pd


def mssv_to_seed(mssv) -> int:
    """Chuyển MSSV thành số nguyên để làm seed.
    Chỉ giữ lại các chữ số, ví dụ 'SV-2021 0001' -> 20210001.
    Nếu không có chữ số nào thì dùng seed = 0."""
    digits = "".join(ch for ch in str(mssv) if ch.isdigit())
    return int(digits) if digits else 0


def generate_data(mssv, n_drivers: int = 20, n_orders: int = 20,
                  area_km: float = 20.0):
    if n_drivers <= 0 or n_orders <= 0:
        raise ValueError("Số tài xế và số đơn hàng phải lớn hơn 0")
    if area_km <= 0:
        raise ValueError("Kích thước khu vực phải lớn hơn 0")

    rng = np.random.default_rng(mssv_to_seed(mssv))

    drivers = pd.DataFrame({
        "driver_id": [f"D{i + 1}" for i in range(n_drivers)],
        "x": rng.uniform(0, area_km, n_drivers),
        "y": rng.uniform(0, area_km, n_drivers),
        "speed_kmh": rng.uniform(25, 50, n_drivers),  # tốc độ trung bình
    })

    orders = pd.DataFrame({
        "order_id": [f"O{i + 1}" for i in range(n_orders)],
        "x": rng.uniform(0, area_km, n_orders),
        "y": rng.uniform(0, area_km, n_orders),
    })

    return drivers, orders


if __name__ == "__main__":
    # Chạy thử: python -m src.data_generator
    d, o = generate_data("20210001", n_drivers=5, n_orders=5)
    print("=== TÀI XẾ ===")
    print(d.round(2))
    print("\n=== ĐƠN HÀNG ===")
    print(o.round(2))