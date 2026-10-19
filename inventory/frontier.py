"""Compare stock-versus-coverage frontiers, not policies at one arbitrary cost ratio."""
import argparse
import numpy as np
import pandas as pd
from .batch import candidates
from .data import load, weekly_demand
from .experiment import evaluation, POLICIES
from .replay import Stock

PENALTIES = (1, 2, 3, 5, 7, 10, 14, 20)


def cohort(sales_zip, inventory_zip, top=0, minimum=5):
    """2024-selected exact T/U pairs; every evaluation series shares the same calendar."""
    pairs = candidates(sales_zip, minimum)[:top or None]
    out = []
    for base, items, _ in pairs:
        try:
            sales, _ = load(sales_zip, inventory_zip, base, items)
            out.append((base, items, weekly_demand(sales, items)))
        except ValueError:
            continue
    return out


def hindsight(demand, penalty, lead=1):
    """Best *constant* stock chosen with 2026 foresight, an infeasible benchmark."""
    y = demand.loc['2026-01-05':].to_numpy(dtype=int)
    peak = y.max(axis=0) * (lead + 1) + 2
    best = None
    warm = 2 * lead + 2
    for t in range(peak[0] + 1):
        for u in range(peak[1] + 1):
            state = Stock(lead, opening=(t, u))
            rows = [state.step(row, (t, u), penalty) for row in y][warm:]
            cost = np.mean([r[0] for r in rows])
            if best is None or cost < best[0]:
                best = (cost, sum(r[1] for r in rows), np.mean([r[3] for r in rows]))
    return best


def run(sales_zip, inventory_zip, top=0, lead=1, penalties=PENALTIES, output=None, minimum=5):
    pairs = cohort(sales_zip, inventory_zip, top, minimum)
    records = []
    for penalty in penalties:
        gathered = {p: [] for p in (*POLICIES, 'Hindsight fixed')}
        for base, items, demand in pairs:
            rows = evaluation(demand, lead, penalty, start_date='2026-01-05')
            if not rows:
                continue
            requests = sum(r['requests'] for r in rows)
            for p in POLICIES:
                gathered[p].append((np.mean([r[p] for r in rows]),
                                    sum(r[p + '_missed'] for r in rows),
                                    np.mean([r[p + '_stock'] for r in rows]), requests))
            cost, missed, stock = hindsight(demand, penalty, lead)
            gathered['Hindsight fixed'].append((cost, missed, stock, requests))
        for policy, vals in gathered.items():
            a = np.asarray(vals)
            records.append({'policy': policy, 'penalty': penalty, 'pairs': len(a),
                            'stock': round(float(a[:, 2].mean()), 3),
                            'coverage': round(float(1 - a[:, 1].sum() / a[:, 3].sum()), 4),
                            'cost': round(float(a[:, 0].mean()), 3)})
    result = pd.DataFrame(records)
    if output:
        result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print('Hindsight uses 2026 test outcomes: not feasible, not a bound on adaptive policies.')
    print('Coverage is of billed purchases; cost uses assumed units, not real dollars.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sales', required=True)
    p.add_argument('--inventory', required=True)
    p.add_argument('--top', type=int, default=0)
    p.add_argument('--minimum', type=int, default=5)
    p.add_argument('--lead', type=int, default=1)
    p.add_argument('--penalties', nargs='+', type=float, default=PENALTIES)
    p.add_argument('--output', default='frontier.csv')
    a = p.parse_args()
    run(a.sales, a.inventory, a.top, a.lead, a.penalties, a.output, a.minimum)
