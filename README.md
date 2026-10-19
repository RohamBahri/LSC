# Inventory decisions with uncertain substitution

A **small exploratory research model**, not an inventory-management system. Starts at phase 3: stochastic demand, customer substitution at stockout, and uncertainty about substitution rates. No private LSC files or assumptions are required.

## Run

```bash
pip install -r requirements.txt
python -m inventory.experiment
pytest -q
```

The experiment compares three policies:

- **Ignore substitution:** choose inventory as if customers never switch products.
- **Nominal:** choose inventory under moderate substitution.
- **Minimax regret:** minimize worst *expected-cost regret* over several demand and substitution scenarios.

All policies choose integer stock quantities `(T, U)` under a shared total capacity, with identical simulated demand paths for each decision within a scenario. Training and evaluation use different random seeds. The scenario-aware oracle is a benchmark, **not a feasible decision under uncertainty**.

## Model

One location and one potentially substitutable product pair. Primary requests are independent Poisson demands. Customers arrive in random order. If the requested product has sold out, a customer may accept the other available product with an assumed probability; otherwise the request goes unmet.

**Objective:** leftover stock cost + unfilled-request cost + accepted-substitution cost. These are reduced-form costs (e.g. lost margin can be included in the unmet penalty), not reported accounting profits. All default quantities and costs are **synthetic**.

Change `CAPACITY`, `COSTS`, `DEMAND`, and scenarios in `inventory/experiment.py`. Enumeration is appropriate for small capacities; this is not a large-scale solver.

We intentionally do **not** assume stockout substitution has been causally identified, or that a partial repair affects future reliability. No inter-base transfers, lead times, dynamic replenishment, or demand learning are modeled yet. The scenario set is a hypothesis, not a statistical confidence region. Minimax regret only protects against the specified scenarios.

## Optional cleaned sales input

```bash
python -m inventory.experiment --sales monthly.csv --base Bonnyville --size 30
```

Provide a CSV with columns `month,base,size,product,units` and products `T` and `U`. The panel must include the entire month calendar, including zero-sales months, and all original transaction exports must already be deduplicated. This simple adapter uses average **observed sales** as a proxy for primary request rates: this can be biased by stockouts and substitution. It is not a demand-identification method. Do not commit private data.

## Layout

- `inventory/model.py` — demand arrivals, substitutions, cost simulation
- `inventory/optimize.py` — exhaustive choices and scenario regret
- `inventory/experiment.py` — assumptions, policies, held-out comparisons
- `inventory/data.py` — optional cleaned monthly CSV input
- `tests/test_model.py` — mechanics checks

## What to investigate

Vary capacity, costs, demand mix, and substitution uncertainty. First ask when uncertainty actually changes the optimal inventory mix and whether robust planning improves held-out decisions. Add another location, lead times, or sequential learning **only after** a finding motivates it.
