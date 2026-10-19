"""Evaluate weekly inventory policies over a cohort selected from 2024 sales only."""
import argparse
import numpy as np
import pandas as pd
from .data import SALES, read_zip, load, weekly_demand
from .experiment import POLICIES, evaluation, summary, interval


def candidates(sales_zip, minimum=5):
    """Select exact T/U SKU pairs by physical base using the *initial* 2024 year."""
    frames = []
    for i, file in enumerate(SALES):
        d = read_zip(sales_zip, file)
        date = pd.to_datetime(d['Posting Date'], format='mixed', errors='coerce')
        if i == 0:
            d = d[date < '2026-03-01']
            date = date.loc[d.index]
        keep = (date >= '2024-01-01') & (date < '2025-01-01')
        keep &= d['Document Type Description'].isin(['Sales Shipment', 'Service Shipment'])
        keep &= d['Item Description'].astype(str).str.startswith('PCP-STATOR ')
        keep &= d['No'].astype(str).str.match(r'^[PS][0-9]') & d.Quantity.gt(0)
        frames.append(d.loc[keep, ['Location Code', 'No', 'Quantity']])
    s = pd.concat(frames, ignore_index=True)
    s['prefix'], s['family'] = s['No'].str[0], s['No'].str[1:]
    counts = s.groupby(['Location Code', 'family', 'prefix']).Quantity.sum().unstack(fill_value=0)
    counts = counts[(counts['P'] >= minimum) & (counts['S'] >= minimum)].copy()
    counts['total'] = counts.P + counts.S
    return [(base, ('P' + family, 'S' + family), int(row.total))
            for (base, family), row in counts.sort_values('total', ascending=False).iterrows()]


def evaluate_all(sales_zip, inventory_zip, top=0, lead=1, penalty=5., mode='none', minimum=5):
    cohort = candidates(sales_zip, minimum)
    if top:
        cohort = cohort[:top]
    records, weekly_changes = [], []
    for base, items, training_units in cohort:
        try:
            sales, ledger = load(sales_zip, inventory_zip, base, items)
            demand = weekly_demand(sales, items)
            rows = evaluation(demand, lead, penalty, mode)
        except ValueError as exc:
            print(f'SKIP {base} {items}: {exc}')
            continue
        item = {'base': base, 'family': items[0][1:], '2024_units': training_units,
                'later_billed_units': int(sum(r['requests'] for r in rows))}
        for p in POLICIES:
            stats = summary(rows, p)
            item.update({p + '_cost': round(stats['cost'], 3),
                         p + '_coverage': round(stats['coverage'], 4),
                         p + '_stock': round(stats['stock'], 3)})
        records.append(item)
        weekly_changes.append([[r[p] - r['Quantile'] for p in POLICIES[1:]] for r in rows])
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError('No eligible pairs were supported by both sales and ledger data')
    print(f'\n{len(result)}/{len(cohort)} eligible real item pairs | lead={lead}, penalty={penalty}, switching={mode}')
    print(result[['base', 'family', '2024_units', 'later_billed_units'] +
                 [p + '_cost' for p in POLICIES]].to_string(index=False))
    print('\nPer-pair median weekly loss:', {p: round(float(result[p + '_cost'].median()), 2) for p in POLICIES})
    print('Median ending stock per pair:', {p: round(float(result[p + '_stock'].median()), 2) for p in POLICIES})
    print('Pooled recorded-purchase coverage:', {
        p: round(sum(r[p + '_coverage'] * r['later_billed_units'] for r in records) /
                 sum(r['later_billed_units'] for r in records), 3) for p in POLICIES})
    paired = np.mean(np.stack(weekly_changes), axis=0)  # Resample common calendar blocks across items.
    for j, p in enumerate(POLICIES[1:]):
        low, high = interval(paired[:, j], np.zeros(len(paired)), seed=2)
        print(f'{p} minus Quantile: mean cost per item-week={paired[:, j].mean():+.3f}, '
              f'synchronized 95% block interval=({low:+.3f}, {high:+.3f})')
    print('Costs are hypothetical. Coverage is of billed purchases, NOT actual service levels.')
    return result


def switching_map(sales_zip, inventory_zip, top=0, lead=1, penalty=5., minimum=5):
    """Maximum possible T/U substitution benefit under the extreme full-switch rule."""
    cohort = candidates(sales_zip, minimum)
    if top:
        cohort = cohort[:top]
    out = []
    for base, items, units in cohort:
        try:
            sales, _ = load(sales_zip, inventory_zip, base, items)
            demand = weekly_demand(sales, items)
            a = summary(evaluation(demand, lead, penalty, 'none'), 'Quantile')
            b = summary(evaluation(demand, lead, penalty, 'both'), 'Quantile')
            out.append((base, items[0][1:], a['cost'] - b['cost'], b['coverage'] - a['coverage']))
        except ValueError:
            continue
    out.sort(key=lambda r: r[2], reverse=True)
    print('Largest hypothetical full-switch gains, Quantile policy; not estimated causal effects')
    for base, family, gain, coverage in out[:15]:
        print(f'{base:<8} {family:<17} cost reduction={gain:6.2f}  coverage increase={coverage:+.1%}')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--top', type=int, default=0, help='0 = all eligible pairs')
    parser.add_argument('--minimum', type=int, default=5, help='Min 2024 sales of each item')
    parser.add_argument('--lead', type=int, default=1)
    parser.add_argument('--penalty', type=float, default=5.)
    parser.add_argument('--mode', choices=('none', 'T_to_U', 'U_to_T', 'both'), default='none')
    parser.add_argument('--switch-map', action='store_true', help='Rank upper-bound substitution benefits')
    parser.add_argument('--output', help='Optional CSV of aggregate pair results')
    a = parser.parse_args()
    result = (switching_map(a.sales, a.inventory, a.top, a.lead, a.penalty, a.minimum)
              if a.switch_map else evaluate_all(a.sales, a.inventory, a.top, a.lead, a.penalty, a.mode, a.minimum))
    if a.output and not a.switch_map:
        result.to_csv(a.output, index=False)
