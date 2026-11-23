import numpy as np
from src.evaluation import total_cost, improvement_pct

def test_total_cost():
    cost = np.array([[1, 2], [3, 4]])
    assert total_cost(cost, [(0, 0), (1, 1)]) == 5

def test_improvement():
    assert improvement_pct(100, 80) == 20.0