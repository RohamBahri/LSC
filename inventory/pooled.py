"""Two pooled rules: fit stock decisions by quantile loss or replay cost.

Tune across all eligible real item pairs using 2025; evaluate only in 2026.
Features are past item demand and past base-wide growth, not generated demand.
"""
import argparse
from itertools import product
import numpy as np
import pandas as pd
from .frontier import cohort
from .replay import Stock

PARAMETERS = list(product((0, .5, 1, 1.5, 2), (.65, .85, 1., 1.2, 1.5), (0, .5)))


def target(series, base_series, week, lead, params):
    """Positive trend-adjusted stock target, updated only from prior weeks."""
    trend, scale, cushion = params
    past = series[max(0, week-52):week]
    recent = past[-13:]
    last4 = past[-4:]
    rates = past.mean(axis=0)
    short = .75 * recent.mean(axis=0) + .25 * last4.mean(axis=0)
    bp = base_series[max(0, week-52):week].sum(axis=1).mean()
    br = base_series[week-13:week].sum(axis=1).mean()
    base_growth = np.clip(br / bp - 1, -1, 2) if bp else 0
    forecast = rates + trend * ((short - rates) + .25 * rates * base_growth)
    return tuple(np.ceil(np.maximum(0, forecast) * (lead + 1) * scale + cushion).astype(int))


def prepare(pairs):
    """Observed weekly series, plus base-wide total rate on the common calendar."""
    x = [(base, data.to_numpy().astype(int)) for base, _, data in pairs]
    bases = {base: np.sum([arr for b, arr in x if b == base], axis=0)
             for base, _ in x}
    return [(base, arr, bases[base]) for base, arr in x]


def score(data, params, penalty, lead, year, method):
    """Evaluate a fixed shared rule on the indicated calendar year."""
    start, stop = (53, 105) if year == 2025 else (105, len(data[0][1]))
    total_loss = 0.
    for base, y, base_y in data:
        q = target(y, base_y, start, lead, params)
        stock = Stock(lead, opening=q)
        for i in range(start, stop):
            if (i - start) % 13 == 0:
                q = target(y, base_y, i, lead, params)
            if method == 'pinball':
                if i % 13 == start % 13 and i + lead < stop:
                    realized = y[i:i+lead+1].sum(axis=0)
                    error = realized - np.asarray(q)
                    alpha = penalty / (1 + penalty)
                    total_loss += np.maximum(alpha * error, (alpha - 1) * error).sum()
            else:
                cost, _, _, _ = stock.step(y[i], q, penalty)
                if i >= start + 2 * lead + 2:
                    total_loss += cost
    return total_loss


def fit(data, penalty, lead=1):
    """One shared parameter tuple; fit with 2025 only, no 2026 outcomes."""
    return {method: min(PARAMETERS, key=lambda theta: score(data, theta, penalty, lead, 2025, method))
            for method in ('pinball', 'cost')}


def evaluation(data, theta, lead=1, penalty=5., year=2026):
    """Frozen shared parameters; on-hand and pipeline carry over within 2026."""
    start, stop = (53, 105) if year == 2025 else (105, len(data[0][1]))
    records = []
    for base, y, base_y in data:
        q = target(y, base_y, start, lead, theta)
        state = Stock(lead, opening=q)
        rows = []
        for i in range(start, stop):
            if (i - start) % 13 == 0:
                q = target(y, base_y, i, lead, theta)
            rows.append((state.step(y[i], q, penalty), int(y[i].sum())))
        rows = rows[2 * lead + 2:]
        records.append(rows)
    return records


def metrics(records):
    """Each pair counts equally for inventory; recorded units pool for coverage."""
    cost = np.mean([np.mean([r[0][0] for r in pair]) for pair in records])
    stock = np.mean([np.mean([r[0][3] for r in pair]) for pair in records])
    sold = sum(sum(r[1] for r in pair) for pair in records)
    missed = sum(sum(r[0][1] for r in pair) for pair in records)
    return round(float(cost), 3), round(float(stock), 3), round(float(1 - missed / sold), 4)


def run(sales_zip, inventory_zip, top=0, lead=1, penalties=(2, 5, 10), output='pooled.csv'):
    data = prepare(cohort(sales_zip, inventory_zip, top))
    rows = []
    for penalty in penalties:
        models = fit(data, penalty, lead)
        for method, params in models.items():
            cost, stock, coverage = metrics(evaluation(data, params, lead, penalty))
            rows.append({'method': 'Pooled ' + method, 'penalty': penalty,
                         'cost': cost, 'stock': stock, 'coverage': coverage,
                         'parameters': str(params), 'pairs': len(data)})
        print(f'penalty={penalty}: {models}')
    result = pd.DataFrame(rows)
    if output:
        result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print('2024: item screen, 2025: pooled tuning, 2026: untouched test.')
    print('Costs assumed; billed purchases treated as requested volumes only for counterfactual replay.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sales', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--top', type=int, default=0)
    p.add_argument('--lead', type=int, default=1)
    p.add_argument('--penalties', nargs='+', type=float, default=[2, 5, 10])
    p.add_argument('--output', default='pooled.csv')
    a = p.parse_args()
    run(a.sales, a.inventory, a.top, a.lead, a.penalties, a.output)
