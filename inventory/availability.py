"""Retrospective inventory accounting check, NOT verified stockout identification."""
import pandas as pd


def stock_bounds(sales, ledger, items, extra=(0, 2, 5)):
    """Minimum nonnegative opening balances under signed transfers minus billed sales.

    The unknown opening balance may exceed this minimum by any amount. Results
    are diagnostics only: posting dates cannot order receipts and sales intraday.
    """
    dates = pd.date_range('2024-01-01', '2026-08-31')
    balances, openings = {}, {}
    for item in items:
        shipped = sales[sales['No'].eq(item)].groupby('date').qty.sum().reindex(dates, fill_value=0)
        movement = ledger[ledger['Item No.'].eq(item)].groupby('date').qty.sum().reindex(dates, fill_value=0)
        net = (movement - shipped).cumsum()
        openings[item] = max(0, int(-net.min()))
        balances[item] = pd.DataFrame({
            'day': dates,
            'sold': shipped.clip(lower=0).to_numpy(),
            'incoming': ledger[(ledger['Item No.'].eq(item)) & (ledger.qty > 0)]
                        .groupby('date').qty.sum().reindex(dates, fill_value=0).to_numpy(),
            'end_min': (net + openings[item]).to_numpy(),
        })
    diagnostics = {}
    for extra_units in extra:
        diagnostics[extra_units] = {
            item: int(((df.end_min + extra_units <= 0) & (df.incoming == 0)).sum())
            for item, df in balances.items()
        }
    return openings, diagnostics
