import numpy as np
import pytest

from src.algorithms import (
    greedy_assign,
    hungarian_custom,
    hungarian_scipy,
    manual_assign,
    random_assign,
)


def _assert_valid_assignment(assignments, n_drivers, n_orders):
    expected_len = min(n_drivers, n_orders)
    assert len(assignments) == expected_len
    drivers = [d for d, _ in assignments]
    orders = [o for _, o in assignments]
    assert len(set(drivers)) == expected_len
    assert len(set(orders)) == expected_len
    assert all(0 <= d < n_drivers for d in drivers)
    assert all(0 <= o < n_orders for o in orders)


def test_known_optimal_solution():
    # Greedy sẽ chọn (0,0)=1 rồi bắt buộc chọn (1,1)=100 -> tổng 101
    # Hungarian tối ưu chọn (0,1)=10 và (1,0)=2 -> tổng 12
    cost = np.array([[1.0, 10.0], [2.0, 100.0]])

    h_assign, h_cost = hungarian_scipy(cost)
    hc_assign, hc_cost = hungarian_custom(cost)
    g_assign, g_cost = greedy_assign(cost)

    _assert_valid_assignment(h_assign, 2, 2)
    _assert_valid_assignment(hc_assign, 2, 2)
    _assert_valid_assignment(g_assign, 2, 2)

    assert np.isclose(h_cost, 12.0)
    assert np.isclose(hc_cost, 12.0)
    assert np.isclose(g_cost, 101.0)
    assert h_cost <= g_cost


@pytest.mark.parametrize("shape", [(5, 5), (4, 7), (8, 5), (15, 15)])
def test_custom_hungarian_matches_scipy_and_beats_baselines(shape):
    rng = np.random.default_rng(2026)
    cost = rng.uniform(1.0, 50.0, size=shape)

    s_assign, s_cost = hungarian_scipy(cost)
    c_assign, c_cost = hungarian_custom(cost)
    g_assign, g_cost = greedy_assign(cost)
    r_assign, r_cost = random_assign(cost, seed=42)
    m_assign, m_cost = manual_assign(cost)

    for assign in (s_assign, c_assign, g_assign, r_assign, m_assign):
        _assert_valid_assignment(assign, shape[0], shape[1])

    assert np.isclose(s_cost, c_cost, atol=1e-9)
    assert s_cost <= g_cost + 1e-9
    assert s_cost <= r_cost + 1e-9
    assert s_cost <= m_cost + 1e-9


def test_random_assign_reproducibility():
    cost = np.random.default_rng(1).uniform(1, 20, size=(6, 6))
    a1, c1 = random_assign(cost, seed=123)
    a2, c2 = random_assign(cost, seed=123)
    assert a1 == a2
    assert c1 == c2