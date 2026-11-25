# Planner-informed network optimization

A research prototype that improves existing production and distribution proposals using historical demand, a learned planner-response model and joint mixed-integer optimization. Production can be reduced or increased; stock in transit and work in progress are tracked explicitly.

## Run

Use Python 3.11+ in a virtual environment. Obtain the authorized sales and inventory archives separately; they are not part of this repository.

```bash
python -m pip install -r requirements.txt
python run_all.py --rebuild --data-dir /path/to/archives
```

The directory must contain `datasets.zip` and `Inventory.zip`. The first command to run the model must include `--rebuild`; subsequent runs may reuse the locally generated arrays and fitted models:

```bash
python run_all.py --window continuous
```

Use `--window short` for the controlled short comparison, or `--original` to include the older production-reduction-only configuration. `refinement_results/` contains the aggregate results, daily decisions and solver checks. The continuous comparison does not reset inventory during the evaluation.

## Layout

- `run_all.py`: single entry point.
- `model/prepare.py`: transaction matching and full-ledger minimum inventory reconstruction.
- `model/forecast.py`, `censored.py`, `learn.py`, `calibrate.py`: demand and planner models.
- `model/policy.py`, `mip.py`: decision model and inventory utilities.
- `model/refined.py`, `evaluate.py`: daily responsive replay and evaluation.
- `model/selected.json`, `pilot_candidate.json`: fixed base settings and recovery-enabled policy.

## Scope and limitations

This is a retrospective research model, not an autonomous production system. Opening inventory is inferred from the full ledger, including later observations. Historical dispatches and future recorded completions stand in for prospective planner proposals. Production and transport lead times, costs and capacity are assumptions. Demand in this implementation is gross-positive booked sales, not document-net demand; the separate simpler evaluation uses a different convention. The historical benchmark uses the same assumed transport delays as the optimized replay. Modeled work in progress and stock are valued using selling-price weights, not actual costs. The recovery configuration was developed after examining extended replay results. No independent prospective validation or verified financial savings are claimed.

Only source code and configuration are tracked. Keep customer records, raw archives, model caches, and detailed outputs private.
