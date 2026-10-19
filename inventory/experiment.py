"""Rolling out-of-time policy evaluation on actual weekly PCP sales."""
import argparse
import numpy as np
from .data import load, weekly_demand, DEFAULT_ITEMS
from .availability import stock_bounds
from .optimize import select
from .replay import Stock

POLICIES = ('Quantile', 'Empirical', 'Robust')


def evaluation(demand, lead=1, penalty=5., mode='none', refresh=13):
    """Quarterly retraining; continuous on-hand stock/pipeline on held-out weeks."""
    y = demand.to_numpy().astype(int)
    dates = demand.index
    start = int(np.searchsorted(dates, np.datetime64('2025-01-06')))
    states = {p: Stock(lead) for p in POLICIES}
    targets = {}
    weekly = []
    for i in range(start, len(y)):
        if (i - start) % refresh == 0:
            train = y[max(0, i-52):i]
            targets = {p: select(train, lead, penalty, p) for p in POLICIES}
        row = {'week': str(dates[i].date())}
        for p in POLICIES:
            cost, missing, switches = states[p].step(y[i], targets[p], penalty, mode)
            row[p] = cost
        weekly.append(row)
    # Cold-start initial inventory is arbitrary; exclude first four evaluation weeks.
    return weekly[4:]


def interval(a, b, seed=1, rep=1000, block=4):
    """Paired four-week moving-block bootstrap on observed cost differences."""
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
    openings, scenarios = stock_bounds(sales, ledger, items)
    print(f'Physical base: {base}; exact SKUs T/U: {items}')
    print(f'Actual billed units {data.sum().tolist()}, weeks {len(data)}; not uncensored requests')
    print(f'Inferred MINIMUM Jan-2024 opening balances: {openings}; actual openings unknown')
    print(f'Possible zero-end-stock days under +0/+2/+5 extra opening units: {scenarios}')
    for penalty in penalties:
        for mode in ('none', 'both'):
            weekly = evaluation(data, lead, penalty, mode)
            print(f'\nHolding:shortage=1:{penalty}, lead={lead} week(s), switching={mode}')
            for p in POLICIES:
                cost = np.mean([r[p] for r in weekly])
                ci = interval([r[p] for r in weekly], [r['Quantile'] for r in weekly]) if p != 'Quantile' else (0., 0.)
                print(f'  {p:<11} mean weekly loss={cost:7.2f}  difference vs Quantile 95% block CI={ci}')
    print('\nCosts and lead time are scenario assumptions. No causal, actual-saving, or policy value claims.')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--base', default='BONPCP')
    parser.add_argument('--items', nargs=2, default=list(DEFAULT_ITEMS), metavar=('T_ITEM', 'U_ITEM'))
    parser.add_argument('--lead', type=int, default=1, help='Assumed delivery lead time in weeks')
    parser.add_argument('--penalties', nargs='+', type=float, default=[2, 5, 10], help='Unmet-sale cost / holding cost')
    a = parser.parse_args()
    run(a.sales, a.inventory, a.base, tuple(a.items), a.lead, tuple(a.penalties))
