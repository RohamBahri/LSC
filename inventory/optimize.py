"""Empirical order-up-to decisions, trained only on past billed demand."""
import numpy as np
from .replay import replay


def choices(history, lead):
    # Candidate upper bounds come from observed lead+review-period demand, not receipts.
    roll = np.array([history[max(0, i-lead):i+1].sum(axis=0)
                     for i in range(len(history))])
    high = roll.max(axis=0).astype(int)
    return [(t, u) for t in range(high[0] + 1) for u in range(high[1] + 1)]


def select(history, lead, penalty, policy):
    history = np.asarray(history, dtype=int)
    if policy == 'Quantile':
        windows = np.array([history[max(0, i-lead):i+1].sum(axis=0)
                            for i in range(len(history))])
        return tuple(np.ceil(np.quantile(windows, penalty / (1 + penalty), axis=0)).astype(int))
    options = choices(history, lead)
    modes = ('none', 'both') if policy == 'Robust' else ('none',)
    scores = np.array([[replay(history, q, lead, penalty, m).mean() for m in modes]
                       for q in options])
    if policy == 'Robust':
        errors = scores - scores.min(axis=0)
        idx = int(np.lexsort((scores.mean(axis=1), errors.max(axis=1)))[0])
    else:
        idx = int(scores[:, 0].argmin())
    return options[idx]
