"""Conventional (non-learning) TE baselines."""

from __future__ import annotations

import networkx as nx
import numpy as np

from sdnte.controller import NetworkState
from sdnte.network import Routing
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
    """Equal-cost multipath: split each pair's demand equally over all min-hop paths.

    TODO (P3, week 2): use nx.all_shortest_paths on topo.graph() and give each
    path a fraction of 1 / number_of_paths.
    """

    name = "ecmp"

    def decide(self, state: NetworkState) -> Routing:
        raise NotImplementedError("ECMP baseline is not implemented yet (week 2, owner P3)")


class GreedyLeastLoaded:
    """Adaptive heuristic: route demands largest-first, each on the path that keeps the
    max utilization along it lowest given the load placed so far.

    TODO (P3, week 2): candidate paths from nx.shortest_simple_paths (K=3..5);
    track residual load per link while placing demands.
    """

    name = "greedy"

    def decide(self, state: NetworkState) -> Routing:
        raise NotImplementedError("Greedy baseline is not implemented yet (week 2, owner P3)")
