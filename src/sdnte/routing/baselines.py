"""Conventional (non-learning) TE baselines."""

from __future__ import annotations

import networkx as nx
import numpy as np

from sdnte.controller import NetworkState
from sdnte.network import Routing
from sdnte.routing.paths import k_shortest_paths
from sdnte.topology import Topology


def shortest_path_routing(topo: Topology, weights: np.ndarray | None = None) -> Routing:
    """One shortest path per reachable OD pair over the links that are up.

    `weights` is a per-link array indexed like `topo.links`; None means hop count.
    Also used by the DRL agent, whose action is a vector of link weights.
    """
    g = topo.graph()
    if weights is not None:
        for _, _, data in g.edges(data=True):
            data["w"] = float(weights[data["index"]])
    routing: Routing = {}
    for s in range(topo.num_nodes):
        if weights is None:
            paths = nx.single_source_shortest_path(g, s)
        else:
            paths = nx.single_source_dijkstra_path(g, s, weight="w")
        for d, path in paths.items():
            if d != s:
                routing[(s, d)] = [(path, 1.0)]
    return routing


class OSPFHop:
    """OSPF with every link cost = 1: minimum-hop routing, blind to load and capacity."""

    name = "ospf_hop"

    def decide(self, state: NetworkState) -> Routing:
        return shortest_path_routing(state.topology)


class OSPFInvCap:
    """OSPF with the Cisco default cost = reference bandwidth / link capacity."""

    name = "ospf_invcap"

    def decide(self, state: NetworkState) -> Routing:
        topo = state.topology
        return shortest_path_routing(topo, topo.capacities.max() / topo.capacities)


class ECMP:
    """Equal-cost multipath: split each pair's demand equally over all min-hop paths."""

    name = "ecmp"

    def decide(self, state: NetworkState) -> Routing:
        topo = state.topology
        g = topo.graph()
        routing: Routing = {}
        for s, d in topo.od_pairs:
            if not nx.has_path(g, s, d):
                continue
            paths = list(nx.all_shortest_paths(g, s, d))
            routing[(s, d)] = [(p, 1.0 / len(paths)) for p in paths]
        return routing


class GreedyLeastLoaded:
    """Adaptive heuristic: place demands largest-first, each on the candidate path
    whose busiest link would be least utilized given the load placed so far.

    Uses the measured traffic matrix, like the DRL agent, but plans one interval
    at a time with no learning.
    """

    name = "greedy"

    def __init__(self, k: int = 4):
        self.k = k

    def decide(self, state: NetworkState) -> Routing:
        topo, tm = state.topology, state.traffic_matrix
        candidates = {
            pair: [(p, topo.path_links(p)) for p in paths]
            for pair, paths in k_shortest_paths(topo, self.k).items()
        }
        loads = np.zeros(topo.num_links)
        routing: Routing = {}
        for s, d in sorted(candidates, key=lambda pair: -tm[pair]):
            demand = tm[s, d]
            best_path, best_links = min(
                candidates[(s, d)],
                key=lambda c: (((loads[c[1]] + demand) / topo.capacities[c[1]]).max(), len(c[1])),
            )
            loads[best_links] += demand
            routing[(s, d)] = [(best_path, 1.0)]
        return routing
