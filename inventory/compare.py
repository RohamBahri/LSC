"""Interpolate frontier coverage at comparable mean inventory (descriptive only)."""
import argparse
import numpy as np
import pandas as pd


def compare(frontier, pooled, levels=(1.5, 2, 2.5, 3)):
    data = pd.concat([frontier.rename(columns={'policy': 'method'}), pooled], ignore_index=True)
    for name, group in data.groupby('method', sort=False):
        curve = group.groupby('stock', as_index=False).coverage.max().sort_values('stock')
        x, y = curve.stock.to_numpy(), curve.coverage.to_numpy()
        values = [f'{np.interp(s, x, y)*100:5.1f}%' if x.min() <= s <= x.max() else '   n/a'
                  for s in levels]
        print(f'{name:<23}' + ''.join(f'{value:>10}' for value in values))
    print('Columns are average ending-stock targets:', levels)
    print('Interpolation between evaluated policies, NOT necessarily feasible intermediate policies.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--frontier', required=True)
    p.add_argument('--pooled', required=True)
    a = p.parse_args()
    compare(pd.read_csv(a.frontier), pd.read_csv(a.pooled))
