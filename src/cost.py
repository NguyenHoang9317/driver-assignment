"""
cost.py
Tính ma trận chi phí cho bài toán phân bổ đơn hàng cho tài xế.

C[i][j] = chi phí để tài xế i nhận đơn hàng j
  - metric = "distance": chi phí là quãng đường (km)
  - metric = "time":     chi phí là thời gian di chuyển (phút)
"""

import numpy as np

METRICS = ("distance", "time")


def distance_matrix(drivers, orders, manhattan: bool = False) -> np.ndarray:
    """Ma trận quãng đường (km), kích thước (số tài xế) x (số đơn).

    manhattan=False: khoảng cách đường chim bay (Euclid).
    manhattan=True:  khoảng cách đi theo ô bàn cờ |dx| + |dy|,
                     giống đi theo đường phố.
    """
    # Dùng broadcasting: cột [:, None] x hàng [None, :] -> ma trận n x m
    dx = drivers["x"].to_numpy()[:, None] - orders["x"].to_numpy()[None, :]
    dy = drivers["y"].to_numpy()[:, None] - orders["y"].to_numpy()[None, :]

    if manhattan:
        return np.abs(dx) + np.abs(dy)
    return np.sqrt(dx ** 2 + dy ** 2)


def cost_matrix(drivers, orders, metric: str = "distance",
                manhattan: bool = False) -> np.ndarray:
    """Ma trận chi phí theo tiêu chí được chọn.

    Tham số:
        drivers:   DataFrame [driver_id, x, y, speed_kmh]
        orders:    DataFrame [order_id, x, y]
        metric:    "distance" (km) hoặc "time" (phút)
        manhattan: True nếu muốn tính khoảng cách Manhattan

    Trả về:
        numpy array shape (len(drivers), len(orders))
    """
    if metric not in METRICS:
        raise ValueError(f"metric phải là một trong {METRICS}, nhận được: {metric!r}")

    dist = distance_matrix(drivers, orders, manhattan)

    if metric == "time":
        speed = drivers["speed_kmh"].to_numpy()[:, None]  # km/h của từng tài xế
        return dist / speed * 60  # giờ -> phút

    return dist


if __name__ == "__main__":
    # Chạy thử:
    from data_generator import generate_data

    d, o = generate_data("20210001", n_drivers=4, n_orders=4)

    print("=== MA TRẬN QUÃNG ĐƯỜNG (km) ===")
    print(cost_matrix(d, o, "distance").round(2))

    print("\n=== MA TRẬN THỜI GIAN (phút) ===")
    print(cost_matrix(d, o, "time").round(2))
