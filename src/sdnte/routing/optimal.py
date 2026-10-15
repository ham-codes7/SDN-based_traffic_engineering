"""LP-optimal TE: the best achievable MLU, used as an upper bound (not deployable live)."""

from __future__ import annotations

from sdnte.controller import NetworkState
from sdnte.network import Routing


class LPOptimal:
    """Multi-commodity flow LP minimizing max link utilization.

    TODO (P3, week 2): solve with scipy.optimize.linprog(method="highs").
    Path-based formulation keeps it small: variables x[p] = fraction of pair (s, d)
    on candidate path p (K-shortest simple paths, K ~ 5) plus U = MLU.
        minimize   U
        subject to sum_p x[p] = 1                                for each OD pair
                   sum_{p uses e} tm[s,d] * x[p] <= U * cap[e]  for each link e
                   x >= 0
    Return the routing as {(s, d): [(path, x[p]), ...]}, dropping near-zero x.
    """

    name = "lp_optimal"

    def decide(self, state: NetworkState) -> Routing:
        raise NotImplementedError("LP-optimal baseline is not implemented yet (week 2, owner P3)")
