"""Sequential customer arrivals: serve the preferred product or try a substitute."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Scenario:
    name: str
    demand: tuple[float, float]       # mean primary requests for T, U per period
    substitution: tuple[float, float] # acceptance rates: T -> U, U -> T


@dataclass(frozen=True)
class Costs:
    leftover: tuple[float, float] = (3, 1)
    unmet: tuple[float, float] = (12, 8)
    switch: tuple[float, float] = (2, 2)


def sample_requests(scenario, periods, seed):
    """Draw primary requests in random arrival order; reuse across stock policies."""
    rng = np.random.default_rng(seed)
    counts = rng.poisson(sum(scenario.demand), periods)
    p_t = scenario.demand[0] / sum(scenario.demand) if sum(scenario.demand) else 0.5
    wanted = np.where(rng.random((periods, max(counts, default=0))) < p_t, 0, 1)
    wanted[np.arange(wanted.shape[1])[None, :] >= counts[:, None]] = -1
    return wanted, rng.random(wanted.shape)


def simulate(stock, scenario, requests, costs=Costs()):
    """Cost per period; substitution happens only when the preferred item is out."""
    wanted, accept = requests
    n = len(wanted)
    remaining = np.tile(np.asarray(stock, dtype=int), (n, 1))
    rows = np.arange(n)
    penalties = np.zeros(n)
    substitution = np.asarray(scenario.substitution)
    unmet = np.asarray(costs.unmet)
    switch_cost = np.asarray(costs.switch)
    for k in range(wanted.shape[1]):
        choice = wanted[:, k].clip(min=0)
        active = wanted[:, k] >= 0
        direct = active & (remaining[rows, choice] > 0)
        swap = (active & ~direct & (remaining[rows, 1 - choice] > 0)
                & (accept[:, k] < substitution[choice]))
        supplied = np.where(direct, choice, np.where(swap, 1 - choice, -1))
        sold = supplied >= 0
        remaining[rows[sold], supplied[sold]] -= 1
        penalties += np.where(active & ~sold, unmet[choice], 0)
        penalties += np.where(swap, switch_cost[choice], 0)
    return penalties + remaining @ np.asarray(costs.leftover)
