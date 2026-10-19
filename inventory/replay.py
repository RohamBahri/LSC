"""Weekly inventory replay: outstanding orders and stock persist across weeks."""


class Stock:
    def __init__(self, lead=1, opening=(0, 0)):
        if lead < 1:
            raise ValueError('Lead time must be at least one week')
        self.onhand = list(map(int, opening))
        self.pipeline = [[0, 0] for _ in range(lead)]

    def step(self, request, target, penalty=5., mode='none'):
        """Receive, order, fulfill: return cost, missed, switched, ending stock."""
        arrived = self.pipeline.pop(0)
        self.onhand = [a + b for a, b in zip(self.onhand, arrived)]
        position = [self.onhand[j] + sum(p[j] for p in self.pipeline) for j in range(2)]
        self.pipeline.append([max(int(target[j]) - position[j], 0) for j in range(2)])
        wanted = list(map(int, request))
        unmet = [max(wanted[j] - self.onhand[j], 0) for j in range(2)]
        self.onhand = [max(self.onhand[j] - wanted[j], 0) for j in range(2)]
        switched = 0
        for origin, destination, rule in ((0, 1, 'T_to_U'), (1, 0, 'U_to_T')):
            if mode in (rule, 'both'):
                n = min(unmet[origin], self.onhand[destination])
                unmet[origin] -= n
                self.onhand[destination] -= n
                switched += n
        missing, holding = sum(unmet), sum(self.onhand)
        return holding + penalty * missing, missing, switched, holding


def replay(demand, target, lead=1, penalty=5., mode='none', warm=True):
    """Replay a candidate decision; remove the first lead+1 burn-in weeks."""
    state = Stock(lead, opening=target if warm else (0, 0))
    costs = [state.step(row, target, penalty, mode)[0] for row in demand]
    return costs[lead + 1:] if len(costs) > lead + 1 else costs
