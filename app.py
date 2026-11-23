import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.data_generator import generate_data, mssv_to_seed
from src.cost import cost_matrix
from src.evaluation import evaluate_all

from src.algorithms.hungarian import solve as hungarian
from src.algorithms.greedy import solve as greedy
from src.algorithms.random_assign import solve as random_assign

SOLVERS = {"Hungarian": hungarian, "Greedy": greedy, "Random": random_assign}

st.set_page_config(page_title="Bài toán gán tài xế - đơn hàng", layout="wide")
st.title("Ghép tài xế với đơn hàng")

with st.sidebar:
    st.header("Cấu hình")
    mssv = st.text_input("MSSV (dùng làm seed)", value="20210001")
    n_drivers = st.slider("Số tài xế", 5, 200, 20)
    n_orders = st.slider("Số đơn hàng", 5, 200, 20)
    criterion = st.radio("Tiêu chí tối ưu", ["Quãng đường (km)", "Thời gian (phút)"])
    manhattan = st.checkbox("Khoảng cách Manhattan", value=False)
    run = st.button("Chạy so sánh", type="primary")

metric = "distance" if criterion.startswith("Quãng") else "time"
unit = "km" if metric == "distance" else "phút"

if run:
    drivers, orders = generate_data(mssv, n_drivers, n_orders)
    cost = cost_matrix(drivers, orders, metric, manhattan)
    res = evaluate_all(cost, SOLVERS, baseline="Random", seed=mssv_to_seed(mssv))
    st.session_state["data"] = (res, drivers, orders, cost, unit)

if "data" in st.session_state:
    res, drivers, orders, cost, unit = st.session_state["data"]

    # 1. Bảng so sánh
    df = pd.DataFrame({
        name: {
            f"Tổng chi phí ({unit})": r["cost"],
            f"TB mỗi đơn ({unit})": r["avg"],
            "Thời gian chạy (s)": r["time"],
            "% cải thiện vs Random": r["improvement"],
        } for name, r in res.items()
    }).T
    st.subheader("Bảng so sánh")
    st.dataframe(df.style.format("{:.4f}"), use_container_width=True)

    # 2. Biểu đồ cột
    st.subheader("Tổng chi phí theo thuật toán")
    st.bar_chart(df[f"Tổng chi phí ({unit})"])

    # 3. Bản đồ ghép cặp
    st.subheader("Bản đồ ghép cặp")
    algo = st.selectbox("Xem ghép cặp của", list(res.keys()))
    pairs = res[algo]["assignment"]

    fig = go.Figure()
    for i, j in pairs:
        fig.add_trace(go.Scatter(
            x=[drivers["x"].iloc[i], orders["x"].iloc[j]],
            y=[drivers["y"].iloc[i], orders["y"].iloc[j]],
            mode="lines", line=dict(width=1, color="gray"), showlegend=False))
    fig.add_trace(go.Scatter(x=drivers["x"], y=drivers["y"], mode="markers",
                             name="Tài xế", text=drivers["driver_id"],
                             marker=dict(size=9, symbol="circle")))
    fig.add_trace(go.Scatter(x=orders["x"], y=orders["y"], mode="markers",
                             name="Đơn hàng", text=orders["order_id"],
                             marker=dict(size=9, symbol="square")))
    fig.update_layout(xaxis_title="x (km)", yaxis_title="y (km)")
    st.plotly_chart(fig, use_container_width=True)

    # 4. Bảng chi tiết
    st.subheader(f"Chi tiết ghép cặp: {algo}")
    detail = pd.DataFrame([
        (drivers["driver_id"].iloc[i], orders["order_id"].iloc[j], cost[i, j])
        for i, j in pairs
    ], columns=["Tài xế", "Đơn hàng", f"Chi phí ({unit})"])
    st.dataframe(detail, use_container_width=True)