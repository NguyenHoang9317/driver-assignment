import streamlit as st

from src.data_generator import generate_data, mssv_to_seed
from src.cost import cost_matrix
from src.evaluation import evaluate_all

try:
    from src.algorithms.hungarian import solve as hungarian
    from src.algorithms.greedy import solve as greedy
    from src.algorithms.random_assign import solve as random_assign
except ImportError:
    from src._stub_local import hungarian, greedy, random_assign

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

# ====== #12 sẽ thêm phần hiển thị ở dưới dòng này ======