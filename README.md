# Inventory decisions using real LSC records

Small exploratory model for **uncertain demand and possible T/U substitution**. No synthetic demand, fabricated sales observations, random substitution rates, or assumed financial costs. The optimization scenarios are **actual historical monthly purchases**, preserving T/U co-movement. The code reads the existing archives locally and does not publish confidential data.

## Run

```bash
pip install -r requirements.txt
python -m inventory.experiment --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip
python -m pytest -q
```

Defaults: physical base **BONPCP**, pump family **043-1200**. Change with `--base LYD --size 015-1400`. You must have both original archives. **Do not commit them to this public repository.**

## What the experiment actually measures

- Six nonoverlapping PCP sales extracts, including the main file through February 2026, avoid the documented February and March export overlap.
- Select actual **T and U stator invoices**, not rotor lines, using physical `Location Code` and the exact nominal-capacity/lift family.
- Use actual positive incoming transfers of matching T/U stators from the serial-level inventory ledger. A receipt is **not** evidence of beginning or ending inventory.
- Compare **three study allocation budgets** drawn from the 25th, 50th, and 75th percentiles of 2024–25 monthly incoming transfers. This is *not a measured capacity limit*; it can be changed once LSC's inventory limits or service requirements are known.
- Select inventory mixes on **January 2024–December 2025** observations, then evaluate on **January–August 2026**, with no future-period model selection.
- Compare a stock mix based on actual historical transfer receipts, a no-switch stochastic allocation, and a **minimax-regret** allocation across possible substitution modes.
- Report average *unserved recorded purchase units* and regret, **not** dollar savings, true lost demand, or realized counterfactual service rates.

No substitution rates are invented. Instead, we evaluate four **logical possibilities**: no switching, T→U only, U→T only, and either direction, all conditional on spare inventory. These are structural *bounds*, not estimates. We cannot identify customers' original first choices or acceptance rates from the sales data alone. This is **sensitivity analysis** using only real demand observations, not proof of causal substitution.

The objective is coverage of historical billed units under a hypothetical monthly stock budget. Because orders can occur throughout a month and local stock carries over, this is **not** a reconstruction of LSC's actual stockout or customer experience. Historical purchased items are a biased proxy for unconstrained requests. It is deliberately a narrow research baseline, not a deployable policy.

## Modules

- `inventory/data.py`: parse actual sales and incoming-transfer archives; build monthly T/U panel
- `inventory/model.py`: count fulfillable units with optional directional substitution
- `inventory/optimize.py`: enumerate inventory mixes and select policies
- `inventory/experiment.py`: historical training and untouched later-period evaluation
- `tests/test_model.py`: tiny arithmetic checks (not experiment inputs)

## Next research step

Inspect sensitivity over **actual** base/size combinations and actual past receipt-volume quantiles. Then use the ledger to reconstruct credible local stock and improve the demand-observation model, explicitly accounting for unknown opening balances and order-driven transfers. Do not estimate substitution causally or claim inventory-policy savings without further evidence.