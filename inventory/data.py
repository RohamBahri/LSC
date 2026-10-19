"""Optional adapter for already cleaned monthly sales (not uncensored demand)."""
import csv
from collections import defaultdict


def observed_rates(path, base, size):
    """CSV columns: month,base,size,product,units; one common calendar for T/U."""
    totals = defaultdict(float)
    months = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["base"] == base and row["size"] == size:
                product = row["product"].upper()
                if product not in ("T", "U"):
                    raise ValueError(f"Unknown product: {product}")
                months.add(row["month"])
                totals[product] += float(row["units"])
    if not months:
        raise ValueError("No matching sales; provide a monthly panel including zero-sales months")
    return tuple(totals[p] / len(months) for p in ("T", "U"))
