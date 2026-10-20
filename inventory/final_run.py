"""Reproducible descriptive benchmark on real LSC records, not an untouched test.

2024 screens item pairs, 2025 fits/checks models, Jan-Aug 2026 replays stock.
External empirical-Bayes and future-rate oracle scripts are NOT included.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .installed_base import panel, compare
from .frontier import run as frontier, PENALTIES
from .pooled import run as pooled
from .batch import switching_map


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def run(sales, inventory, output='results/final', quick=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    leads = (1,) if quick else (1, 2)
    penalties = (2,) if quick else PENALTIES
    manifest = {
        'classification': 'Exploratory, repeatedly viewed 2025/2026 periods; no new holdout',
        'sales_sha256': digest(sales), 'inventory_sha256': digest(inventory),
        'selection_year': 2024, 'model_validation_year': 2025,
        'stock_replay': 'Jan-Aug 2026; incomplete Q3, no full Q3 evaluation',
        'cohort_minimum_2024_T_and_U': [5, 1], 'leads_weeks': leads,
        'shortage_penalties': penalties, 'holding_penalty': 1,
        'switching': 'none, except a separately labeled full-switching sensitivity',
        'not_included': ['empirical Bayes from external script',
                         'future-rate oracle from external script'],
        'quick_test': quick,
    }
    (out / 'protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for min_units, label in ((5, '26'), (1, '126')):
        print(f'\n## 2025 well-history forecast comparison: {label} cohort', flush=True)
        frame = panel(sales, min_units, top=2 if quick else 0)
        models = compare(frame)
        frame.to_csv(out / f'features_{label}.csv', index=False)
        observed = frame[frame.year == 2025].next13.to_numpy()
        pd.DataFrame([{'model': name, 'wape': np.abs(pred - observed).sum() / observed.sum(),
                       'bias': (pred - observed).sum() / observed.sum()}
                      for name, pred in models]).to_csv(out / f'well_models_{label}.csv', index=False)
        for lead in leads:
            print(f'\n## Stock-coverage frontiers: {label} pairs, lead={lead}', flush=True)
            frontier(sales, inventory, top=2 if quick else 0, lead=lead,
                     minimum=min_units, penalties=penalties,
                     output=out / f'frontier_{label}_lead{lead}.csv')
            print(f'\n## Shared pooled policy: {label} pairs, lead={lead}', flush=True)
            pooled(sales, inventory, top=2 if quick else 0, lead=lead,
                   minimum=min_units, penalties=penalties,
                   output=out / f'pooled_{label}_lead{lead}.csv')
    print('\n## Complete switching (optimistic, not identified)', flush=True)
    changes = switching_map(sales, inventory, top=2 if quick else 0, lead=1, penalty=5, minimum=5)
    pd.DataFrame(changes, columns=['base', 'item_family', 'cost_upper_bound',
                                    'recorded_coverage_upper_bound']).to_csv(out / 'switching_bounds.csv', index=False)
    print(f'\nDone. Send the entire {out} folder, including protocol.json and the run log.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sales', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', default='results/final')
    parser.add_argument('--quick', action='store_true', help='2 pairs, one lead, one penalty; smoke test only')
    args = parser.parse_args()
    run(args.sales, args.inventory, args.output, args.quick)
