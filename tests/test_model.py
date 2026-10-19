"""Hand-checked mechanics; no synthetic demand in research experiments."""
import numpy as np
from inventory.replay import Stock, replay
from inventory.optimize import select
from inventory.experiment import interval


def test_lead_and_carryover():
    s = Stock(lead=1)
    assert s.step((2, 0), (2, 0), 5) == (10., 2, 0)
    assert s.step((1, 0), (2, 0), 5) == (1., 0, 0)
    assert s.step((0, 0), (2, 0), 5) == (1., 0, 0)


def test_substitution_is_not_primary_demand():
    s = Stock(lead=1)
    s.step((0, 0), (0, 2))
    assert s.step((2, 0), (0, 2), mode='both') == (0., 0, 2)


def test_minimization_and_interval():
    h = np.array([[0, 2], [0, 1], [2, 0], [1, 1]])
    for p in ('Quantile', 'Empirical', 'Robust'):
        assert len(select(h, 1, 5, p)) == 2
    assert interval([2]*10, [1]*10) == (1., 1.)
    assert len(replay(h, (1, 1))) == 4
