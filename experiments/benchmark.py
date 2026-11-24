import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lưu ảnh, không mở cửa sổ
import matplotlib.pyplot as plt

from src.data_generator import generate_data
from src.cost import cost_matrix
from src.evaluation import evaluate_all

from src.algorithms import get_assignments


from src.algorithms import PAIR_ALGORITHMS, BASELINE

OUT_DIR = "experiments"

UNITS = {"distance": "km", "time": "phút"}

def run_benchmark(sizes=(10, 20, 50, 100, 200), seeds=range(1, 11),
                  metric="distance"):
    rows = []
    for n in sizes:
        for s in seeds:
            drivers, orders = generate_data(str(s), n_drivers=n, n_orders=n)
            cost = cost_matrix(drivers, orders, metric)
            res = evaluate_all(cost, SOLVERS, baseline="Random", seed=s)
            for name, r in res.items():
                rows.append(dict(n=n, seed=s, algo=name, cost=r["cost"],
                                 time=r["time"], improvement=r["improvement"]))
    return pd.DataFrame(rows)


def plot_runtime(df):
    t = df.groupby(["n", "algo"])["time"].mean().unstack()
    ax = t.plot(marker="o", logy=True)
    ax.set_xlabel("n (số tài xế = số đơn)")
    ax.set_ylabel("Thời gian chạy trung bình (s)")
    ax.set_title("Thời gian chạy theo n")
    ax.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "runtime.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def plot_cost(df):
    c = df.groupby(["n", "algo"])["cost"].mean().unstack()
    ax = c.plot(marker="o")
    ax.set_xlabel("n")
    ax.set_ylabel("Tổng chi phí trung bình (km)")
    ax.set_title("Tổng chi phí theo n (trung bình nhiều seed)")
    ax.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "cost.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = run_benchmark()
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    plot_runtime(df)
    plot_cost(df)

    summary = df.groupby(["n", "algo"])[["cost", "time", "improvement"]] \
                .agg(["mean", "std"]).round(4)
    print(summary)