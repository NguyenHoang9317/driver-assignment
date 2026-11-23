import time
import numpy as np

def total_cost(cost, assignment):
    return float(sum(cost[i, j] for i, j in assignment))

def avg_cost(cost, assignment):
    return total_cost(cost, assignment) / max(len(assignment), 1)

def improvement_pct(baseline_cost, algo_cost):
    """% cải thiện so với baseline (dương = tốt hơn baseline)."""
    if baseline_cost == 0:
        return 0.0
    return (baseline_cost - algo_cost) / baseline_cost * 100

def run_and_measure(solver, cost, **kwargs):
    t0 = time.perf_counter()
    assignment = solver(cost, **kwargs)
    elapsed = time.perf_counter() - t0
    return assignment, elapsed

def evaluate_all(cost, solvers: dict, baseline="Random", seed=0):
    """solvers = {"Hungarian": f1, "Greedy": f2, "Random": f3}"""
    results = {}
    for name, fn in solvers.items():
        kw = {"seed": seed} if name == "Random" else {}
        asg, t = run_and_measure(fn, cost, **kw)
        results[name] = {"assignment": asg, "cost": total_cost(cost, asg),
                         "avg": avg_cost(cost, asg), "time": t}
    base = results[baseline]["cost"]
    for r in results.values():
        r["improvement"] = improvement_pct(base, r["cost"])
    return results