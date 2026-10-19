# Weekly inventory decisions on real LSC records

A lean **research replay**, not an LSC deployment tool. It uses actual weekly PCP sales from `datasets.zip`, with the inventory ledger in `Inventory.zip` for an independent opening-balance audit. No synthetic demand, invented switching probabilities, or fabricated financial returns.

## Run

```bash
pip install -r requirements.txt
python -m inventory.experiment --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip
python -m pytest -q
```

Defaults: physical location `BONPCP`, exact T/U item numbers `P043120MN1D312EP` and `S043120MN1D312EP` (both family 043-1200, MN1 elastomer). Choose another **compatible** pair with `--base LYD --items T_ITEM U_ITEM`. Item number matters: different elastomers are **not** pooled. Never commit confidential ZIPs to this public repository.

## What is evaluated

- **Observed input:** deduplicated positive PCP shipment quantities, by complete Monday–Sunday week, January 2024–August 2026. Exports are read without the documented February/March overlap; final partial week is excluded. These are recorded purchases, **not known primary demand**.
- **Decision:** two integer order-up-to levels, one for each item. At the start of each week, prior orders arrive, a new replenishment order is placed, and recorded purchases are fulfilled. Unused inventory and orders in transit carry across weeks.
- **Lead time:** an assumed fixed 1 week (`--lead 2` tests two weeks). We have not estimated delivery lead times from order and receipt timestamps. There are no order-pulled emergency transfers or within-week replenishments.
- **Cost sensitivity:** average end-of-week units held × 1 plus missed recorded purchase units × an assumed penalty (2, 5, or 10). These are **cost ratios**, not LSC dollars. Change with `--penalties 3 6`.
- **Substitution sensitivity:** no switching versus full two-way switching whenever spare stock exists. This is a scenario envelope, **not a causal estimate, actual physical interchangeability, or observed customer behavior**. No switching should be the primary interpretation.
- **Methods:** an empirical lead-plus-review-period **Quantile** baseline; an **Empirical** order-up-to level minimizing past replay cost; **Robust** minimax expected-cost regret across the two switching extremes. Search bounds come from the actual past demand windows, not receipt flow.
- **Evaluation:** all policies re-select levels every 13 weeks based on only the previous 52 weeks; stock and outstanding orders are carried continuously through the out-of-time test (2025–August 2026). The first four evaluation weeks are omitted to reduce arbitrary zero-start effects. Results include paired four-week moving-block bootstrap intervals for weekly cost differences versus Quantile.

### Important limitations

1. The ledger records signed movements but lacks verified January 2024 opening stocks. The code reports **minimum nonnegative opening balances** inferred retrospectively and checks additional opening balances of +0/+2/+5 units. Those are *diagnostics*, not evidence of actual zero-stock or verified lost sales. Posting dates cannot determine whether receipts preceded same-day invoices.
2. Historical sales depend on inventory and substitution. A new policy could change customer behavior; the replay does **not** recover that counterfactual and cannot establish actual cost savings or service improvement. Its held-out cost is a **model-based proxy** on historical billed quantities.
3. Order arrival timing, storage cost, unmet-demand cost, and substitution rules are not observed. Do not claim estimates of these quantities. Lack of cost data is handled using explicitly varied **relative** costs.
4. Comparisons concern one location and a *single exact T/U SKU pair*. Broader evidence requires rerunning on additional pairs and learning the demand/stock observation process.

## Modules

- `inventory/data.py` — read and deduplicate actual sales and ledger records, weekly exact-SKU purchases
- `inventory/availability.py` — audited lower-bound stock reconstruction and opening-balance sensitivity
- `inventory/replay.py` — weekly carryover, replenishment lead time, shortages, switch sensitivity
- `inventory/optimize.py` — simple and robust weekly order-up-to selection
- `inventory/experiment.py` — rolling real-data evaluation and paired uncertainty intervals
- `tests/test_model.py` — hand-checked mechanics

## Next question

Do differences in **demand timing** and **replenishment lead time** matter more than possible T/U substitution? If so, extend this weekly replay to pooled policies learned across many sparse item/location series, with inventory cost as the objective. Actual cost, fulfillment, and request data would improve identification, but no more complicated model is needed to begin testing that hypothesis.