"""Flow-level data-plane model: routes a traffic matrix and measures the result.

The model is analytical rather than packet-level:
- link load = sum of demand fractions routed over the link
- utilization = load / capacity; MLU = max utilization over links that are up
- drop rate on an overloaded link = 1 - capacity / load
- link delay = propagation + M/M/1 sojourn time (1/mu) / (1 - rho), rho capped at 0.99
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sdnte.topology import Topology

# A routing maps each (src, dst) pair to one or more paths, each with the
# fraction of that pair's demand it carries. Fractions per pair sum to 1.
Routing = dict[tuple[int, int], list[tuple[list[int], float]]]

PACKET_BITS = 12_000  # 1500-byte packets
MAX_QUEUE_UTIL = 0.99


@dataclass
class Metrics:
    mlu: float
    mean_util: float
    loss_pct: float
    throughput_ratio: float
    avg_delay_ms: float
    delay_std_ms: float
    offered_mbps: float
    delivered_mbps: float
    link_util: np.ndarray = field(repr=False)

    def summary(self) -> dict[str, float]:
        """Scalar metrics only, for tables and CSVs."""
        return {k: v for k, v in self.__dict__.items() if k != "link_util"}


def _flatten(topo: Topology, routing: Routing, tm: np.ndarray):
    """Routing as flat arrays: flow per path, and (link index, path id) per hop."""
    flows, hop_links, hop_paths = [], [], []
    for (s, d), splits in routing.items():
        demand = tm[s, d]
        if demand <= 0:
            continue
        for path, frac in splits:
            idx = topo.path_links(path)
            hop_paths.extend([len(flows)] * len(idx))
            hop_links.extend(idx)
            flows.append(demand * frac)
    return np.array(flows), np.array(hop_links, dtype=int), np.array(hop_paths, dtype=int)


def link_loads(topo: Topology, routing: Routing, tm: np.ndarray) -> np.ndarray:
    """Offered load in Mbps on every directed link."""
    flows, hop_links, hop_paths = _flatten(topo, routing, tm)
    if not len(flows):
        return np.zeros(topo.num_links)
    return np.bincount(hop_links, weights=flows[hop_paths], minlength=topo.num_links)


def evaluate(topo: Topology, routing: Routing, tm: np.ndarray) -> Metrics:
    """Route traffic matrix `tm` (Mbps, shape N x N) with `routing` and measure it."""
    flows, hop_links, hop_paths = _flatten(topo, routing, tm)
    n_paths = len(flows)
    caps = topo.capacities
    up = topo.up
    loads = np.bincount(hop_links, weights=flows[hop_paths], minlength=topo.num_links) if n_paths else np.zeros(topo.num_links)

    util = np.where(up, loads / caps, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        drop = np.where(loads > 0, np.clip(1.0 - caps / loads, 0.0, 1.0), 0.0)
    # Traffic sent onto a failed link is lost entirely.
    drop = np.where(up, drop, np.where(loads > 0, 1.0, 0.0))

    service_rate = caps * 1e6 / PACKET_BITS  # packets/s
    queue_ms = 1e3 / service_rate / (1.0 - np.minimum(util, MAX_QUEUE_UTIL))
    link_delay = topo.delays + queue_ms

    # Demand for pairs missing from `routing` (unreachable) counts as offered but lost.
    offered = float(tm.sum() - np.trace(tm))
    # A path delivers its flow times the product of (1 - drop) over its links.
    log_survive = np.log(np.clip(1.0 - drop[hop_links], 1e-12, 1.0))
    survive = np.exp(np.bincount(hop_paths, weights=log_survive, minlength=n_paths))
    delivered = float((flows * survive).sum())
    path_delays = np.bincount(hop_paths, weights=link_delay[hop_links], minlength=n_paths)

    if n_paths:
        avg_delay = float(np.average(path_delays, weights=flows))
        delay_std = float(np.sqrt(np.average((path_delays - avg_delay) ** 2, weights=flows)))
    else:
        avg_delay = delay_std = 0.0

    return Metrics(
        mlu=float(util.max()) if up.any() else 0.0,
        mean_util=float(util[up].mean()) if up.any() else 0.0,
        loss_pct=max(0.0, 100.0 * (1.0 - delivered / offered)) if offered > 0 else 0.0,
        throughput_ratio=min(1.0, delivered / offered) if offered > 0 else 1.0,
        avg_delay_ms=avg_delay,
        delay_std_ms=delay_std,
        offered_mbps=offered,
        delivered_mbps=delivered,
        link_util=util,
    )
