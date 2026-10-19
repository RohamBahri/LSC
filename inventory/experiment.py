"""Run a small research experiment: simple vs nominal vs robust stock choices."""
import argparse
import numpy as np
from .model import Scenario, Costs
from .optimize import allocations, best_regret, expected_costs
from .data import observed_rates


CAPACITY = 12
COSTS = Costs()                 # Illustrative; change to test economic assumptions.
DEMAND = (6.0, 2.0)            # Mean primary requests; synthetic by default.


def run(demand=DEMAND, capacity=CAPACITY, train=3000, test=6000):
    scenarios = [
        Scenario("No substitution", demand, (0.0, 0.0)),
        Scenario("Moderate", demand, (0.4, 0.1)),
        Scenario("Strong", demand, (0.8, 0.3)),
        Scenario("High T demand", (1.3 * demand[0], demand[1]), (0.6, 0.2)),
        Scenario("High U demand", (demand[0], 1.5 * demand[1]), (0.6, 0.2)),
    ]
    choices = allocations(capacity)
    fitted = expected_costs(choices, scenarios, COSTS, train, seed=10)
    policies = {
        "Ignore substitution": int(np.argmin(fitted[:, 0])),
        "Nominal": int(np.argmin(fitted[:, 1])),
        "Minimax regret": best_regret(fitted),
    }
    heldout = expected_costs(choices, scenarios, COSTS, test, seed=1000)
    oracle = heldout.min(axis=0)  # Scenario-aware benchmark, not a feasible policy.

    print("Decisions:", {name: choices[i] for name, i in policies.items()})
    print("Costs: unmet T/U=12/8, leftover T/U=3/1, switching=2; units are illustrative")
    print(f"{'Scenario':<19} {'Oracle':>9}" + "".join(f" {s:>20}" for s in policies))
    for j, s in enumerate(scenarios):
        print(f"{s.name:<19} {oracle[j]:9.2f}" +
              "".join(f" {heldout[i, j]:20.2f}" for i in policies.values()))
    print("Worst regret:" + "".join(
        f" {max(heldout[i] - oracle):20.2f}" for i in policies.values()))
    print("Mean regret: " + "".join(
        f" {np.mean(heldout[i] - oracle):20.2f}" for i in policies.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sales", help="Cleaned monthly sales CSV (optional)")
    parser.add_argument("--base")
    parser.add_argument("--size")
    args = parser.parse_args()
    if args.sales and not (args.base and args.size):
        parser.error("--sales requires --base and --size")
    means = observed_rates(args.sales, args.base, args.size) if args.sales else DEMAND
    if args.sales:
        print("WARNING: historical sales are a proxy for primary demand, not an estimate of lost requests.")
    run(demand=means)
