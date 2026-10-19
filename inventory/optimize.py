"""Enumerate feasible two-product inventory levels and compare decisions."""
import numpy as np
from .model import sample_requests, simulate


def allocations(capacity):
    return [(t, u) for t in range(capacity + 1)
            for u in range(capacity - t + 1)]


def expected_costs(choices, scenarios, costs, periods=3000, seed=1):
    """Paired Monte Carlo: all decisions face identical demand in each scenario."""
    result = np.empty((len(choices), len(scenarios)))
    for j, scenario in enumerate(scenarios):
        requests = sample_requests(scenario, periods, seed + j)
        for i, stock in enumerate(choices):
            result[i, j] = simulate(stock, scenario, requests, costs).mean()
    return result


def best_regret(matrix):
    """Minimize maximum expected regret across the specified scenarios."""
    regret = matrix - matrix.min(axis=0)
    return int(np.lexsort((matrix.mean(axis=1), regret.max(axis=1)))[0])
