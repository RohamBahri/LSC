"""Rolling inventory-policy evaluation on real, intermittent weekly sales."""
import argparse
import numpy as np
from .data import load, weekly_demand, DEFAULT_ITEMS
from .availability import stock_bounds
from .optimize import select_all
from .replay import Stock

POLICIES = ('Quantile', 'Empirical', 'Robust')


def evaluation(demand, lead=1, penalty=5., mode='none', refresh=13, start_date='2025-01-06'):
    """Quarterly fitting on preceding 52 weeks, held-out replay from 2025 onward."""
    y = demand.to_numpy().astype(int)
    start = int(np.searchsorted(demand.index, np.datetime64(start_date)))
    states = None
    weekly = []
    for i in range(start, len(y)):
        if (i - start) % refresh == 0:
            targets = select_all(y[max(0, i - 52):i], lead, penalty)
            if states is None:
                states = {p: Stock(lead, opening=targets[p]) for p in POLICIES}
        row = {'week': str(demand.index[i].date()), 'requests': int(y[i].sum())}
        for p in POLICIES:
            cost, missed, switched, stock = states[p].step(y[i], targets[p], penalty, mode)
            row[p] = cost
            row[p + '_missed'] = missed
            row[p + '_stock'] = stock
            row[p + '_switch'] = switched
        weekly.append(row)
    return weekly[2 * lead + 2:]  # Comparable post-warm-up weeks, across all policies.


def summary(rows, policy):
    """Replay coverage refers to *recorded purchases*, never unidentified lost demand."""
    count = sum(r['requests'] for r in rows)
    missed = sum(r[policy + '_missed'] for r in rows)
    return {'cost': np.mean([r[policy] for r in rows]),
            'coverage': (count - missed) / count if count else np.nan,
            'stock': np.mean([r[policy + '_stock'] for r in rows]),
            'missed': missed}


def interval(a, b, seed=1, rep=1000, block=4):
    """Paired, moving-block bootstrap interval for the average weekly cost difference."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    n = len(d)
    if n < block:
        return (float('nan'), float('nan'))
    means = []
    for _ in range(rep):
        starts = rng.integers(0, n - block + 1, size=(n + block - 1) // block)
        inds = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means.append(d[inds].mean())
    return tuple(np.quantile(means, [.025, .975]))


def run(sales_zip, inventory_zip, base='BONPCP', items=DEFAULT_ITEMS, lead=1, penalties=(2, 5, 10)):
    sales, ledger = load(sales_zip, inventory_zip, base, items)
    data = weekly_demand(sales, items)
    openings, diagnostics = stock_bounds(sales, ledger, items)
    print(f'{base}, exact T/U SKUs: {items}; billed units: {data.sum().tolist()}')
    print(f'Minimum opening-stock inference: {openings}; zero-day sensitivity: {diagnostics}')
    for penalty in penalties:
        for mode in ('none', 'both'):
            rows = evaluation(data, lead, penalty, mode)
            print(f'\nUnmet:holding={penalty}:1; lead={lead} week(s); switching={mode}')
            for p in POLICIES:
                s = summary(rows, p)
                ci = interval([r[p] for r in rows], [r['Quantile'] for r in rows]) if p != 'Quantile' else (0., 0.)
                print(f'{p:<10} loss={s["cost"]:.2f}  recorded coverage={s["coverage"]:.1%}  '
                      f'end stock={s["stock"]:.1f}  difference CI=({ci[0]:.2f},{ci[1]:.2f})')
    print('Counterfactual replay of billed sales: NOT actual fill rates or savings.')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--base', default='BONPCP')
    parser.add_argument('--items', nargs=2, default=list(DEFAULT_ITEMS))
    parser.add_argument('--lead', type=int, default=1)
    parser.add_argument('--penalties', nargs='+', type=float, default=[2, 5, 10])
    a = parser.parse_args()
    run(a.sales, a.inventory, a.base, tuple(a.items), a.lead, tuple(a.penalties))
