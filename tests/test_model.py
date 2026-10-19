"""Tiny hand-checkable arithmetic tests; not research data or experiment inputs."""
from inventory.model import fulfill
from inventory.optimize import allocations, loss_matrix, policies


def test_no_switch():
    assert fulfill((2, 0), (1, 3)) == (1, 0, 3)


def test_directions():
    assert fulfill((0, 3), (2, 1), "T_to_U") == (3, 2, 0)
    assert fulfill((3, 0), (1, 2), "U_to_T") == (3, 2, 0)
    assert fulfill((0, 3), (2, 1), "none") == (1, 0, 2)


def test_choices_and_regret():
    choices = allocations(2)
    assert choices == [(0, 2), (1, 1), (2, 0)]
    losses = loss_matrix(choices, [(2, 0), (0, 2)])
    assert losses.shape == (3, 4)
    result = policies(choices, losses, [(1, 1), (1, 1)])
    assert all(0 <= i < len(choices) for i in result.values())
