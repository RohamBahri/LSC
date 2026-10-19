import numpy as np
from inventory.model import Scenario, Costs, sample_requests, simulate
from inventory.optimize import allocations, best_regret, expected_costs


def test_substitution_and_cost():
    requests = (np.array([[0]]), np.array([[0.0]]))
    yes = Scenario("yes", (1, 0), (1, 0))
    no = Scenario("no", (1, 0), (0, 0))
    assert simulate((0, 1), yes, requests)[0] == 2  # T request uses U
    assert simulate((0, 1), no, requests)[0] == 13  # lost T + leftover U


def test_arrival_order_and_conservation():
    requests = (np.array([[0, 0, 1]]), np.zeros((1, 3)))
    scenario = Scenario("x", (2, 1), (1, 0))
    assert simulate((1, 1), scenario, requests)[0] == 10  # one unmet U, one switch


def test_zero_demand_and_capacity():
    choices = allocations(3)
    assert len(choices) == 10 and all(sum(q) <= 3 for q in choices)
    scenario = Scenario("empty", (0, 0), (0, 0))
    requests = sample_requests(scenario, 5, 1)
    assert np.all(simulate((2, 1), scenario, requests) == 7)


def test_regret_selection():
    matrix = np.array([[1, 9], [5, 5], [9, 1]])
    assert best_regret(matrix) == 1
    result = expected_costs([(0, 0), (1, 0)], [Scenario("x", (1, 1), (0, 0))], Costs(), 10, 1)
    assert result.shape == (2, 1)
