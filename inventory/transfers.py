"""Audit serial-linked ERP posting gaps; NOT order-to-delivery lead times."""
import argparse
import pandas as pd
from .data import LEDGER, read_zip


def posting_gaps(inventory_zip):
    ledger = read_zip(inventory_zip, LEDGER)
    x = ledger[ledger['Entry Type Description'].eq('Transfer')].copy()
    x['day'] = pd.to_datetime(x['Posting Date'], format='%m/%d/%Y', errors='coerce')
    keys = ['Order No.', 'Serial No.', 'Item No.']
    source = (x[x['Location Code'].eq('EDM') & x.Quantity.lt(0)][keys + ['day']]
              .drop_duplicates(keys).rename(columns={'day': 'source_day'}))
    destination = (x[x.Quantity.gt(0) & x['Location Code'].isin(['ATH', 'BONPCP', 'LYD', 'MAC'])]
                   [keys + ['Location Code', 'day']].rename(columns={'day': 'dest_day'}))
    merged = destination.merge(source, on=keys)
    merged['posting_gap'] = (merged.dest_day - merged.source_day).dt.days
    return merged


def report(inventory_zip):
    x = posting_gaps(inventory_zip)
    print('Serial-matched Edmonton transfer posting-date gaps (NOT shipping lead times)')
    print(x.groupby('Location Code').posting_gap.agg(['count', 'median', 'mean', 'min', 'max']).round(2).to_string())
    print('Same-day:', int(x.posting_gap.eq(0).sum()), '/', len(x),
          '; negative:', int(x.posting_gap.lt(0).sum()))
    print('Order-placement and physical dispatch/receipt times are not recorded reliably.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    a = parser.parse_args()
    report(a.inventory)
