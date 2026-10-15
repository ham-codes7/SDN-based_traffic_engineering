"""LP-optimal TE: the best achievable MLU, used as an upper bound (not deployable live)."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from sdnte.controller import NetworkState
from sdnte.network import Routing
from sdnte.routing.paths import k_shortest_paths

# Tiny cost on hop count so that, among routings with the same MLU, the LP prefers
# short paths instead of arbitrary detours.
HOP_PENALTY = 1e-4


class LPOptimal:
    """Path-based multi-commodity flow LP minimizing max link utilization (MLU).

    Variables: x[p] = fraction of pair (s, d) on candidate path p, and U = MLU.
        minimize   U + eps * sum_p hops(p) * share(s, d) * x[p]
        subject to sum_p x[p] = 1                                for each OD pair
                   sum_{p uses e} tm[s,d] * x[p] <= U * cap[e]  for each link e
                   x >= 0
    Candidate paths are the k shortest simple paths per pair; with k = 8 this is
    within a fraction of a percent of the unrestricted optimum on our topologies.
    """

    name = "lp_optimal"

    def __init__(self, k: int = 8):
        self.k = k

    def decide(self, state: NetworkState) -> Routing:
        topo, tm = state.topology, state.traffic_matrix
        candidates = k_shortest_paths(topo, self.k)
        pairs = [pair for pair in candidates if tm[pair] > 0]
        paths = [(pair, p) for pair in pairs for p in candidates[pair]]
        n_x = len(paths)
        u_col = n_x  # index of U

        total = sum(tm[pair] for pair in pairs)
        c = np.zeros(n_x + 1)
        c[u_col] = 1.0
        for j, (pair, p) in enumerate(paths):
            c[j] = HOP_PENALTY * (len(p) - 1) * tm[pair] / total

        a_ub = lil_matrix((topo.num_links, n_x + 1))
        a_eq = lil_matrix((len(pairs), n_x + 1))
        pair_row = {pair: i for i, pair in enumerate(pairs)}
        for j, (pair, p) in enumerate(paths):
            a_eq[pair_row[pair], j] = 1.0
            for e in topo.path_links(p):
                a_ub[e, j] = tm[pair]
        a_ub[:, u_col] = -topo.capacities.reshape(-1, 1)

        res = linprog(
            c,
            A_ub=a_ub.tocsr(),
            b_ub=np.zeros(topo.num_links),
            A_eq=a_eq.tocsr(),
            b_eq=np.ones(len(pairs)),
            bounds=[(0, None)] * (n_x + 1),
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        routing: Routing = {}
        for j, (pair, p) in enumerate(paths):
            if res.x[j] > 1e-6:
                routing.setdefault(pair, []).append((p, float(res.x[j])))
        for pair, splits in routing.items():  # renormalize after dropping tiny shares
            s = sum(f for _, f in splits)
            routing[pair] = [(p, f / s) for p, f in splits]
        return routing
