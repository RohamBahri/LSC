"""Real-data decision experiment: 2024-25 training, 2026 out-of-time evaluation."""
import argparse
import numpy as np
from .data import monthly_history
from .model import MODES
from .optimize import allocations, loss_matrix, policies


def run(sales, inventory, base="BONPCP", size="043-1200"):
    history = monthly_history(sales, inventory, base, size)
    train, test = history.loc[:"2025-12"], history.loc["2026-01":]

    print(f"Physical location: {base} | pump family: {size}")
    print(f"Observed billed units: train={int(train[['T','U']].sum().sum())}, "
          f"test={int(test[['T','U']].sum().sum())}; months={len(train)}+{len(test)}")
    print("Historical receipts are NOT stock on hand; billed units are NOT uncensored demand.")
    print("Substitution modes are possibility bounds, NOT measured acceptance rates.")
    results = {}
    for quantile in (0.25, 0.50, 0.75):
        # Study budget from actual training receipts, NOT physical storage capacity.
        capacity = max(1, int(np.quantile(train[["T_received", "U_received"]].sum(axis=1), quantile)))
        choices = allocations(capacity)
        fitted = loss_matrix(choices, train[["T", "U"]].to_numpy())
        selection = policies(choices, fitted, train[["T_received", "U_received"]].to_numpy())
        holdout = loss_matrix(choices, test[["T", "U"]].to_numpy())
        print(f"\nHistorical receipt-volume quantile {quantile:.0%}: budget={capacity} units")
        print("Decisions:", {name: choices[i] for name, i in selection.items()})
        print(f"{'Mode':<12}" + "".join(f"{name:>18}" for name in selection))
        for j, mode in enumerate(MODES):
            print(f"{mode:<12}" + "".join(f"{holdout[i, j]:18.2f}" for i in selection.values()))
        results[quantile] = {name: (choices[i], holdout[i].copy()) for name, i in selection.items()}
    return history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", required=True, help="Local datasets.zip (never committed)")
    parser.add_argument("--inventory", required=True, help="Local Inventory.zip (never committed)")
    parser.add_argument("--base", default="BONPCP", help="Physical Location Code")
    parser.add_argument("--size", default="043-1200", help="Nominal capacity-lift family")
    args = parser.parse_args()
    run(args.sales, args.inventory, args.base, args.size)
