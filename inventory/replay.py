"""Weekly order-up-to inventory replay with lost sales and pipeline carryover."""
import numpy as np


class Stock:
    def __init__(self, lead=1):
        if lead < 1:
            raise ValueError('Lead time must be at least one week')
        self.onhand = np.zeros(2, dtype=int)
        self.pipeline = np.zeros((lead, 2), dtype=int)

    def step(self, request, target, penalty=5., mode='none'):
        """Receive old orders, place new ones, then fill recorded weekly purchases.

        'both' is a full-switching upper bound, NOT a measured acceptance rate.
        """
        self.onhand += self.pipeline[0].copy()
        self.pipeline[:-1] = self.pipeline[1:].copy()
        self.pipeline[-1] = 0
        self.pipeline[-1] += np.maximum(np.asarray(target) - self.onhand - self.pipeline.sum(axis=0), 0)
        demand = np.asarray(request, dtype=int)
        directly = np.minimum(self.onhand, demand)
        self.onhand -= directly
        left = demand - directly
        switched = 0
        if mode in ('T_to_U', 'both'):
            n = min(left[0], self.onhand[1]); self.onhand[1] -= n; left[0] -= n; switched += n
        if mode in ('U_to_T', 'both'):
            n = min(left[1], self.onhand[0]); self.onhand[0] -= n; left[1] -= n; switched += n
        return float(self.onhand.sum() + penalty * left.sum()), int(left.sum()), switched


def replay(demand, target, lead=1, penalty=5., mode='none'):
    state = Stock(lead)
    return np.array([state.step(row, target, penalty, mode)[0] for row in demand])
