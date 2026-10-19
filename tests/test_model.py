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


def test_pooled_uses_only_previous_weeks():
    from inventory.pooled import target
    y = np.ones((110, 2), dtype=int)
    theta = (.5, 1., 0.)
    first = target(y, y, 105, 1, theta)
    altered = y.copy()
    altered[105:] *= 100
    assert target(altered, altered, 105, 1, theta) == first


def test_stock_and_hindsight():
    import pandas as pd
    from inventory.frontier import hindsight
    from inventory.experiment import evaluation
    dates = pd.date_range('2024-01-01', '2026-08-24', freq='W-MON')
    weekly = pd.DataFrame(np.ones((len(dates), 2), dtype=int), index=dates)
    rows = evaluation(weekly, start_date='2026-01-05')
    assert rows and rows[0]['week'] >= '2026-01-05'
    assert len(hindsight(weekly, 5)) == 3


def test_install_history_origin_not_leaking():
    import pandas as pd
    from inventory.installed_base import origin_fits_year
    assert origin_fits_year(pd.Timestamp('2025-10-06'))
    assert not origin_fits_year(pd.Timestamp('2024-12-30'))
