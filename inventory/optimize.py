"""Small exhaustive inventory optimization over actual historical months."""
import numpy as np
from .model import MODES, fulfill


def allocations(capacity):
    return [(t, capacity - t) for t in range(capacity + 1)]


def loss_matrix(choices, months):
    """Average unserved *recorded* units for each choice and substitution possibility."""
    return np.array([[np.mean([fulfill(stock, demand, mode)[2] for demand in months])
                      for mode in MODES] for stock in choices])


def policies(choices, losses, receipts):
    """Historical receipt mix, no-switch optimum, and min-max mode regret."""
    mean_receipts = np.asarray(receipts, dtype=float).sum(axis=0)
    if not mean_receipts.sum():
        raise ValueError("No training-period receipts")
    target_t = choices[0][1] * mean_receipts[0] / mean_receipts.sum()
    historical = min(range(len(choices)), key=lambda i: abs(choices[i][0] - target_t))
    direct = int(np.argmin(losses[:, 0]))
    regret = losses - losses.min(axis=0)
    robust = int(np.lexsort((losses.mean(axis=1), regret.max(axis=1)))[0])
    return {"Receipt mix": historical, "No-switch": direct, "Minimax regret": robust}
