# Inventory decisions under uncertainty — real LSC records

A small research replay of **weekly order-up-to inventory policies**. Demand inputs are real recorded PCP shipments, never artificially generated. The optimization question is general; LSC supplies the historical observations. This is **not** a deployable recommendation system or a causal substitution estimate.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python -m inventory.batch --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip
```

The cohort is selected **using 2024 sales only**: matching T/U stators with the same exact suffix (including elastomer), at one physical location, at least 5 units of **each** product. All eligible pairs are evaluated; use `--top 12` for a shorter run. Sales are deduplicated across the known February/March file overlaps. No proprietary data is committed.

```bash
# Substitution sensitivity: extreme assumption, NOT observed response probabilities.
python -m inventory.batch --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip --mode both

# Where complete switching could matter most, as an optimistic scenario bound.
python -m inventory.batch --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip --switch-map

# Audit serial-linked posting-date gaps (NOT physical replenishment lead times).
python -m inventory.transfers --inventory /path/to/Inventory.zip

# Single exact item pair, opening-stock diagnostics and cost-ratio sensitivity.
python -m inventory.experiment --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip
```

## Model

Each week, previously ordered inventory arrives, the company orders up to a target inventory position, and then billed quantities are used as a proxy for requests. Unsold stock and open orders persist. Targets are selected quarterly from the preceding 52 weeks, using **no future sales**. Compare:

- **Quantile:** observed lead-plus-review-period demand quantile (simple benchmark).
- **Empirical:** lowest historical weekly holding-plus-unmet cost in a warm-started replay.
- **Robust:** smallest worst regret over two possible switching behaviors: none, or full two-way T/U substitution.

Replay uses a lead time of 1 week by default (`--lead 2` for sensitivity). Each item starts with the candidate's inventory level in training; initial evaluation stock is set from its first policy target, then carried forward. We omit the initial warm-up weeks. Report modeled weekly cost, **coverage of recorded purchases**, mean ending stock, and synchronized paired moving-block confidence intervals across item pairs. Holding cost is one normalized unit; unmet sales are penalized by an **assumed** ratio, e.g. `--penalty 5` in batch.

### Interpretation limits

1. Historical **sales are not unconstrained customer requests**: stockouts and substitution may have censored or changed them. Replaying a new inventory policy over billed purchases is a **model-based scenario test**, not an unbiased estimate of real lost demand, fill rate, savings or causal policy effects.
2. Substitution is unverified. Full switching gives an **optimistic modeling extreme**, not a validated physical repair option. No actual customer-choice probability is estimated.
3. Product costs, shortage costs, physical order lead times, and initial on-hand stock are not reliably identified. Cost ratios and lead times are explicit sensitivity assumptions. The ledger's serial-linked origin/destination **posting** dates are often the same day or inconsistent, so they cannot determine physical shipping and order lead times.
4. The opening-stock audit reports the **smallest nonnegative inferred balance** and sensitivity to additional unobserved initial stock. Zero balances inferred this way are not verified stockouts.
5. The cohort and 2024 eligibility screen are fixed before the later evaluation period, but the exploratory cost scenarios were chosen by researchers. Intervals quantify temporal variation conditional on this replay and selected cohort; they are not deployment confidence intervals.

## Modules

- `data.py`: original sales and serial inventory archives, exact product IDs, weekly purchases
- `availability.py`: opening-stock uncertainty audit
- `replay.py`: lead time, continuous inventory, substitution, and cost
- `optimize.py`: three compact, per-pair replenishment policies
- `experiment.py`: single-pair rolling evaluation and diagnostics
- `batch.py`: all eligible real item pairs, pooled comparisons, full-switch sensitivity
- `transfers.py`: serial-linked **posting** gaps, not delivery lead time
- `tests/`: hand-verifiable checks


## Inventory–coverage frontiers and pooled learning

Compare **2026 held-out stock–coverage curves**, not a single shortage penalty:

```bash
python -m inventory.frontier --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip --output frontier.csv
python -m inventory.pooled --sales /path/to/datasets.zip --inventory /path/to/Inventory.zip --output pooled.csv
python -m inventory.compare --frontier frontier.csv --pooled pooled.csv
```

- `frontier.py` sweeps shortage penalties for Quantile, Empirical, Robust, and a **fixed** hindsight benchmark chosen using all 2026 demand. The benchmark is unavailable to real decision makers and **not** an upper bound for all adaptive policies.
- `pooled.py` fits **one shared low-dimensional trend/scale/stock-cushion rule** across the selected item pairs. Inputs are the exact item's trailing 4/13/52-week billed rates and its physical base's recent growth. Two variants choose shared parameters on **2025 only**: one minimizes next-horizon quantile (pinball) loss; the other minimizes the full inventory replay loss. The rule is deliberately much simpler than a general pooled ML model.
- The cohort is chosen from **2024**. Every method is evaluated on **January–August 2026**, with targets recomputed quarterly from prior observations. 2026 is **never used to fit pooled parameters**. Replays start from the first target and discard the same opening weeks.
- `compare.py` interpolates observed curve points at common average ending-stock levels. **Interpolated values are descriptive and need not be achievable by a deterministic policy.** All policies use real billed volumes as a demand proxy; service coverage is not a verified customer fill rate.

Initial 26-pair held-out result: at average ending stock of ~3 units, pooled rules cover ~86.6% of recorded purchases versus Quantile ~85.5%; at lower inventories the pooled rules do not outperform Quantile. No established pooled-learning advantage yet. The gap to the hindsight fixed-stock result cannot automatically be closed by a learnable policy.

**Literature warning:** Gupta and Kallus, *Data Pooling in Stochastic Optimization*, *Management Science* 68(3), 1595–1615 (2022), DOI `10.1287/mnsc.2020.3933`, already develops decision-data pooling for many small stochastic programs and evaluates retail inventory. Pooling by itself is **not a novelty claim**.

## What we learned, and the next question

In an exploratory run of 26 pairs (2024 T/U volumes at least five each), **Quantile covers more recorded purchases**, while Empirical and Robust show no clear advantage in modeled weekly costs at an assumed 5:1 shortage-to-holding ratio. Full interchangeability changes modeled coverage much more than policy choice, but it is **not identified** from the available data.

The current pooled trend/scale prototype is a first comparison, not evidence of superiority. Further research should investigate whether forecast bias, censored primary demand, or changes in the installed base explain the gap between out-of-sample policies and the test-informed fixed-stock benchmark. Any improvement must be measured across later time periods under the same cost/lead assumptions. Do not infer actual service performance without verified stock and primary demand.