"""Load exact PCP items at their physical location from the original archives."""
from functools import lru_cache
from zipfile import ZipFile
import pandas as pd

SALES = [
    'datasets/PCP Sales.csv',
    'datasets/PCP Sales for UoA March to April 2026.csv',
    'datasets/PCP_Sales_UoA_2026-05.csv',
    'datasets/PCP_Sales_UoA_202606.csv',
    'datasets/PCP_Sales_2026-07.csv',
    'datasets/PCP_Sales_2026-08.csv',
]
LEDGER = 'Inventory/PCP non-inv transactions ItemLedger_UoA_2024_2026SEP25.csv'
DEFAULT_ITEMS = ('P043120MN1D312EP', 'S043120MN1D312EP')


@lru_cache(maxsize=10)
def read_zip(archive, name):
    with ZipFile(archive) as z:
        return pd.read_csv(z.open(name), low_memory=False)


def load(sales_zip, inventory_zip, base='BONPCP', items=DEFAULT_ITEMS):
    """Return dated billed quantities and signed transfer movements, exact item IDs.

    Billing is a proxy for requested demand; transfers aren't all available stock.
    """
    if len(set(items)) != 2:
        raise ValueError('Select two different item numbers: T followed by U')
    frames = []
    for i, name in enumerate(SALES):
        d = read_zip(sales_zip, name)
        d['date'] = pd.to_datetime(d['Posting Date'], format='mixed', errors='coerce').dt.normalize()
        if i == 0:  # Later export replaces the overlapping March 1-10 records.
            d = d[d.date < '2026-03-01']
        frames.append(d[d['Location Code'].eq(base) & d['No'].isin(items)])
    sales = pd.concat(frames, ignore_index=True)
    sales = sales[sales['Document Type Description'].isin(['Service Shipment', 'Sales Shipment'])].copy()
    sales['qty'] = pd.to_numeric(sales.Quantity, errors='coerce').fillna(0)
    sales = sales[sales.date.between('2024-01-01', '2026-08-31')]
    ledger = read_zip(inventory_zip, LEDGER)
    ledger = ledger[ledger['Location Code'].eq(base) & ledger['Item No.'].isin(items)].copy()
    ledger['date'] = pd.to_datetime(ledger['Posting Date'], format='%m/%d/%Y', errors='coerce')
    ledger['qty'] = pd.to_numeric(ledger.Quantity, errors='coerce').fillna(0)
    ledger = ledger[ledger.date.between('2024-01-01', '2026-08-31')]
    if sales.empty or ledger.empty or any((sales['No'] == x).sum() == 0 for x in items):
        raise ValueError('No usable sales/ledger history for both selected item numbers at this location')
    return sales, ledger


def weekly_demand(sales, items):
    """Complete Monday-start calendar of recorded purchases, including zero weeks."""
    days = pd.date_range('2024-01-01', '2026-08-24', freq='W-MON')  # Last complete week.
    d = sales[(sales.qty > 0) & (sales.date <= '2026-08-30')].copy()
    d['week'] = d.date - pd.to_timedelta(d.date.dt.weekday, unit='D')
    return (d.pivot_table(index='week', columns='No', values='qty', aggfunc='sum')
            .reindex(index=days, columns=items).fillna(0).astype(int))
