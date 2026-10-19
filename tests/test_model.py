"""Small hand-checked mechanics tests; research runs always use real demand."""
from inventory.replay import Stock
from inventory.optimize import select_all, windows
from inventory.experiment import interval, summary
import numpy as np


def test_lead_and_carryover():
    s = Stock(lead=1)
    assert s.step((2, 0), (2, 0), 5) == (10, 2, 0, 0)
    assert s.step((1, 0), (2, 0), 5) == (1, 0, 0, 1)
    assert s.step((0, 0), (2, 0), 5) == (1, 0, 0, 1)


def test_substitution():
    s = Stock(lead=1)
    s.step((0, 0), (0, 2))
    assert s.step((2, 0), (0, 2), mode='both') == (0, 0, 2, 0)


def test_warm_start():
    s = Stock(lead=1, opening=(3, 1))
    assert s.step((1, 0), (3, 1)) == (3, 0, 0, 3)


def test_selection_and_interval():
    y = np.array([[0, 2], [0, 1], [2, 0], [1, 1]])
    assert windows(y, 1).shape == (3, 2)
    q = select_all(y, 1, 5)
    assert set(q) == {'Quantile', 'Empirical', 'Robust'}
    assert interval([2]*10, [1]*10) == (1., 1.)


def test_coverage():
    r = [{'requests': 3, 'Quantile': 7, 'Quantile_missed': 1, 'Quantile_stock': 2}]
    assert summary(r, 'Quantile')['coverage'] == 2 / 3
