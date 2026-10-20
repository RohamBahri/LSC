"""Cheap 2025 test: do previous well-linked stator bills predict future sales?

Recorded bills are NOT verified installations or a complete active installed base.
No 2026 observations are used for training or evaluation.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from .data import SALES, read_zip

HORIZON = 13


def origin_fits_year(date):
    return (date + pd.Timedelta(weeks=HORIZON - 1)).year == date.year


def panel(archive, minimum=1, top=0):
    """Build 2024/25 quarterly origins for pairs selected from 2024 sales."""
    source = read_zip(archive, SALES[0])
    dates = pd.to_datetime(source['Posting Date'], errors='coerce')
    keep = ((dates >= '2024-01-01') & (dates < '2025-01-01') &
            source['Document Type Description'].isin(['Sales Shipment', 'Service Shipment']) &
            source['Item Description'].str.startswith('PCP-STATOR ', na=False) &
            source['No'].astype(str).str.match(r'^[PS][0-9]') & source.Quantity.gt(0))
    sample = source.loc[keep, ['Location Code', 'No', 'Quantity']].copy()
    sample['type'], sample['family'] = sample['No'].str[0], sample['No'].str[1:]
    counts = sample.groupby(['Location Code', 'family', 'type']).Quantity.sum().unstack(fill_value=0)
    counts = counts[(counts['P'] >= minimum) & (counts['S'] >= minimum)].copy()
    counts['total'] = counts.P + counts.S
    pairs = [(base, ('P' + family, 'S' + family), int(row.total))
             for (base, family), row in counts.sort_values('total', ascending=False).iterrows()][:top or None]
    keys = [(base, item) for base, items, _ in pairs for item in items]
    accepted = set(keys)
    s = source.copy()
    s['date'] = pd.to_datetime(s['Posting Date'], errors='coerce').dt.normalize()
    s = s[(s.date < '2026-01-01') & s.date.notna() &
          s['Document Type Description'].isin(['Sales Shipment', 'Service Shipment']) &
          s['Item Description'].str.startswith('PCP-STATOR ', na=False) &
          s.Quantity.gt(0)].copy()
    s['item'] = s['No'].astype(str)
    s['base'] = s['Location Code'].astype(str)
    s['well'] = s.Company.astype(str) + ':' + s.UWI.astype(str).str.strip().str.upper()
    s['valid_well'] = s.UWI.astype(str).str.strip().str.upper().str.fullmatch('[A-Z0-9]{15,20}')
    s['week'] = s.date - pd.to_timedelta(s.date.dt.weekday, unit='D')

    weekly = s.groupby(['week', 'base', 'item']).Quantity.sum()
    weeks = pd.date_range('2023-01-02', '2025-12-29', freq='W-MON')
    matrix = pd.DataFrame({key: weekly.reindex(
        pd.MultiIndex.from_product([weeks, [key[0]], [key[1]]]), fill_value=0
    ).to_numpy() for key in keys}, index=weeks)

    # Track each well's most recent LSC stator bill. Ambiguous same-day
    # multi-SKU records are excluded; these are not verified installations.
    events = s[s.valid_well & s['Installation Type'].isin(['NWC', 'REPLSI', 'REPCOM'])]
    unique = events.groupby(['well', 'date']).filter(lambda g: g['item'].nunique() == 1)
    unique = unique.sort_values('date').drop_duplicates(['well', 'date'], keep='last')
    event_list = list(unique[['well', 'date', 'base', 'item']].itertuples(index=False, name=None))
    last, position, rows = {}, 0, []
    for date in weeks[(weeks >= '2024-01-01') & (weeks <= '2025-10-06')]:
        while position < len(event_list) and event_list[position][1] < date:
            well, when, base, item = event_list[position]
            last[well] = (when, base, item)
            position += 1
        age_counts = {key: [0, 0, 0] for key in keys}
        for when, base, item in last.values():
            key = (base, item)
            if key in accepted:
                age = (date - when).days / 7
                age_counts[key][0 if age < 52 else 1 if age < 156 else 2] += 1
        ix = weeks.get_loc(date)
        if ix + HORIZON > len(weeks) or not origin_fits_year(date):
            continue
        first = pd.Timestamp(f'{date.year}-01-01')
        monday = first + pd.Timedelta(days=(7 - first.weekday()) % 7)
        if (date - monday).days // 7 % HORIZON:
            continue
        base_recent = {base: float(matrix.loc[weeks[ix-13:ix],
                       [k for k in keys if k[0] == base]].to_numpy().sum())
                       for base in {b for b, _ in keys}}
        for base, item in keys:
            y = matrix[(base, item)].to_numpy()
            rows.append({'year': date.year, 'date': date.date(), 'base': base,
                         'product': item[0], 'item': item,
                         'last4': y[ix-4:ix].sum(), 'last13': y[ix-13:ix].sum(),
                         'last52': y[ix-52:ix].sum(), 'base13': base_recent[base],
                         'well_lt52': age_counts[(base, item)][0],
                         'well_52_156': age_counts[(base, item)][1],
                         'well_gt156': age_counts[(base, item)][2],
                         'next13': y[ix:ix+HORIZON].sum()})
    return pd.DataFrame(rows)


def compare(df):
    """Fit on 2024 origins and evaluate 2025, without 2026 targets."""
    train, valid = df[df.year == 2024], df[df.year == 2025]
    common = ['last4', 'last13', 'last52', 'base13']
    exposure = ['well_lt52', 'well_52_156', 'well_gt156']
    design = pd.get_dummies(df[['base', 'product']], dtype=float)
    observed = valid.next13.to_numpy()
    results = [('Trailing 52 weeks', valid.last52.to_numpy() / 4)]
    for name, features in [('Pooled sales only', common),
                           ('Pooled + well history', common + exposure)]:
        x = pd.concat([np.log1p(df[features].astype(float)), design], axis=1)
        xtrain, xvalid = x.loc[train.index], x.loc[valid.index]
        scale = StandardScaler().fit(xtrain)
        learner = PoissonRegressor(alpha=1.0, max_iter=300).fit(scale.transform(xtrain), train.next13)
        prediction = learner.predict(scale.transform(xvalid))
        results.append((name, prediction))
    print(f'2024 training: {len(train)} item-origin rows; 2025 evaluation: {len(valid)}; '
          f'products: {df[["base","item"]].drop_duplicates().shape[0]}')
    print(f'{"Method":<27} {"MAE/quarter":>13} {"WAPE":>10} {"Bias":>10}')
    for name, pred in results:
        err = pred - observed
        print(f'{name:<27} {np.abs(err).mean():13.3f} '
              f'{np.abs(err).sum()/observed.sum():10.1%} {err.sum()/observed.sum():10.1%}')
    print('2025 is exploratory validation, not a pristine publication holdout.')
    print('Well features represent last OBSERVED billing, not verified installations.')
    return results


def run(sales, minimum=1, top=0, output='well_features_2025.csv'):
    df = panel(sales, minimum, top)
    if output:
        df.to_csv(output, index=False)
        print('Feature panel saved:', output)
    compare(df)
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sales', required=True, help='Local datasets.zip')
    ap.add_argument('--minimum', type=int, default=1)
    ap.add_argument('--top', type=int, default=0)
    ap.add_argument('--output', default='well_features_2025.csv')
    args = ap.parse_args()
    run(args.sales, args.minimum, args.top, args.output)
