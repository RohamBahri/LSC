"""Order-up-to decisions on actual historical weeks; no artificial demand."""
import numpy as np
from .replay import replay


def windows(history, lead):
    y = np.asarray(history, dtype=int)
    if len(y) <= lead:
        raise ValueError('Need more observed weeks than lead time')
    return np.array([y[i - lead:i + 1].sum(axis=0) for i in range(lead, len(y))])


def select_all(history, lead=1, penalty=5.):
    """Fit all three policies once on the same trailing history."""
    y = np.asarray(history, dtype=int)
    w = windows(y, lead)
    critical = penalty / (1 + penalty)
    quantile = tuple(np.ceil(np.quantile(w, critical, axis=0)).astype(int))
    high = w.max(axis=0)
    choices = [(t, u) for t in range(high[0] + 1) for u in range(high[1] + 1)]
    scores = np.array([[np.mean(replay(y, q, lead, penalty, mode))
                        for mode in ('none', 'both')] for q in choices])
    regrets = scores - scores.min(axis=0)
    robust = int(np.lexsort((scores.mean(axis=1), regrets.max(axis=1)))[0])
    return {'Quantile': quantile, 'Empirical': choices[int(scores[:, 0].argmin())],
            'Robust': choices[robust]}


