"""Candidate path sets shared by the multipath baselines (greedy, LP-optimal)."""

from __future__ import annotations

from itertools import islice

import networkx as nx

from sdnte.topology import Topology

_cache: dict[tuple, dict[tuple[int, int], list[list[int]]]] = {}


def k_shortest_paths(topo: Topology, k: int) -> dict[tuple[int, int], list[list[int]]]:
    """Up to k loop-free paths per reachable OD pair, fewest hops first.

    Cached per (topology, set of links that are up, k), so recomputation only
    happens after a link failure.
    """
    key = (topo.name, topo.up.tobytes(), k)
    if key not in _cache:
        g = topo.graph()
        paths = {}
        for s, d in topo.od_pairs:
            if nx.has_path(g, s, d):
                paths[(s, d)] = list(islice(nx.shortest_simple_paths(g, s, d), k))
        _cache[key] = paths
    return _cache[key]
