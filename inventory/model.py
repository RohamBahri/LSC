"""Exact fulfillment counts for observed requests; no synthetic demand generation."""

MODES = ("none", "T_to_U", "U_to_T", "both")


def fulfill(stock, requests, mode="none"):
    """Return (served, substitutions, unserved) for one observed T/U month.

    The mode specifies *possible* switching, not an estimated behavior probability.
    Recorded sales are used as a proxy for requests; neither mode is causal evidence.
    """
    t, u = map(int, requests)
    x, y = map(int, stock)
    direct = min(t, x) + min(u, y)
    switch_t = min(max(t - x, 0), max(y - u, 0)) if mode in ("T_to_U", "both") else 0
    switch_u = min(max(u - y, 0), max(x - t, 0)) if mode in ("U_to_T", "both") else 0
    switched = switch_t + switch_u
    return direct + switched, switched, t + u - direct - switched
