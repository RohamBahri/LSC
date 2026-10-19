"""Read actual PCP sales and physical-location receipts from the LSC archives."""
from zipfile import ZipFile
import pandas as pd

SALES = [
    "datasets/PCP Sales.csv",
    "datasets/PCP Sales for UoA March to April 2026.csv",
    "datasets/PCP_Sales_UoA_2026-05.csv",
    "datasets/PCP_Sales_UoA_202606.csv",
    "datasets/PCP_Sales_2026-07.csv",
    "datasets/PCP_Sales_2026-08.csv",
]
LEDGER = "Inventory/PCP non-inv transactions ItemLedger_UoA_2024_2026SEP25.csv"


def read_zip(path, member):
    with ZipFile(path) as archive:
        return pd.read_csv(archive.open(member), low_memory=False)


def product_lines(df, description="Item Description"):
    parts = df[description].astype("string").str.extract(r"PCP-STATOR\s+([TU])\s+(\d{3}-\d{4})\b")
    return df.assign(product=parts[0], size=parts[1])


def monthly_history(sales_zip, inventory_zip, base="BONPCP", size="043-1200"):
    """Return monthly billed T/U quantities and incoming T/U transfers, 2024-2026.

    Receipts are *not* on-hand stock. Billed sales are *not* uncensored requests.
    """
    records = []
    for i, member in enumerate(SALES):
        df = read_zip(sales_zip, member)
        df["date"] = pd.to_datetime(df["Posting Date"], format="mixed", errors="coerce")
        if i == 0:                       # Main file overlaps the separate March extract
            df = df[df.date < "2026-03-01"]
        records.append(df)
    sales = product_lines(pd.concat(records, ignore_index=True))
    sales["quantity"] = pd.to_numeric(sales["Quantity"], errors="coerce")
    sales = sales[(sales["Location Code"] == base) & (sales["size"] == size) &
                  sales["product"].isin(["T", "U"]) & (sales.quantity > 0) &
                  sales["Document Type Description"].isin(["Sales Shipment", "Service Shipment"])]
    sales["month"] = sales.date.dt.to_period("M")
    billed = sales.pivot_table(index="month", columns="product", values="quantity",
                               aggfunc="sum", fill_value=0)

    ledger = product_lines(read_zip(inventory_zip, LEDGER))
    ledger["quantity"] = pd.to_numeric(ledger["Quantity"], errors="coerce")
    ledger["date"] = pd.to_datetime(ledger["Posting Date"], format="mixed", errors="coerce")
    ledger = ledger[(ledger["Location Code"] == base) & (ledger["size"] == size) &
                    ledger["product"].isin(["T", "U"]) & (ledger.quantity > 0) &
                    (ledger["Entry Type Description"] == "Transfer")]
    ledger["month"] = ledger.date.dt.to_period("M")
    received = ledger.pivot_table(index="month", columns="product", values="quantity",
                                  aggfunc="sum", fill_value=0)

    months = pd.period_range("2024-01", "2026-08", freq="M")
    billed = billed.reindex(index=months, columns=["T", "U"], fill_value=0)
    received = received.reindex(index=months, columns=["T", "U"], fill_value=0)
    if billed.to_numpy().sum() == 0 or received.to_numpy().sum() == 0:
        raise ValueError("No matching billed products or receipt transfers; check physical base and pump family")
    result = billed.rename(columns={"T": "T", "U": "U"})
    result["T_received"], result["U_received"] = received["T"], received["U"]
    return result.astype(int)
