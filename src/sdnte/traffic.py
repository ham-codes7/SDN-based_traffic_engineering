"""Traffic matrix (TM) generators.

A traffic matrix `tm[s, d]` is the demand in Mbps from switch s to switch d during
one control interval. `generate_sequence` returns one TM per interval, shape (T, N, N).
"""

from __future__ import annotations

import numpy as np

from sdnte.topology import Topology

PATTERNS = ("uniform", "gravity", "hotspot", "bursty")


def base_matrix(topo: Topology, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Normalized (sums to 1) N x N demand shape for a traffic pattern."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown traffic pattern {pattern!r}; choose from {PATTERNS}")
    n = topo.num_nodes
    if pattern == "uniform":
        tm = rng.uniform(0.5, 1.5, (n, n))
    else:
        # Gravity model: a node sends/receives in proportion to its attached capacity.
        node_cap = np.zeros(n)
        np.add.at(node_cap, [l.src for l in topo.links], topo.capacities)
        w_out = node_cap * rng.lognormal(0.0, 0.3, n)
        w_in = node_cap * rng.lognormal(0.0, 0.3, n)
        tm = np.outer(w_out, w_in)
    np.fill_diagonal(tm, 0.0)

    if pattern == "hotspot":
        # A few destinations (e.g. data centres) receive half of all traffic.
        hot = rng.choice(n, size=max(1, n // 6), replace=False)
        hot_sum = tm[:, hot].sum()
        tm[:, hot] *= (tm.sum() - hot_sum) / hot_sum
    return tm / tm.sum()


def generate_sequence(
    topo: Topology, pattern: str, steps: int, load: float, seed: int = 0
) -> np.ndarray:
    """Time-varying TM sequence, shape (steps, N, N), scaled to the requested load.

    Every TM is the pattern's base shape times a diurnal cycle (one period per
    sequence) and per-entry lognormal noise. "bursty" adds random 3-6x spikes.
    """
    rng = np.random.default_rng(seed)
    n = topo.num_nodes
    base = base_matrix(topo, pattern, rng)
    t = np.arange(steps)
    diurnal = 1.0 + 0.2 * np.sin(2 * np.pi * t / max(steps, 1))
    seq = base[None] * diurnal[:, None, None] * rng.lognormal(0.0, 0.1, (steps, n, n))
    if pattern == "bursty":
        spikes = rng.random((steps, n, n)) < 0.05
        seq[spikes] *= rng.uniform(3.0, 6.0, spikes.sum())
    seq[:, np.arange(n), np.arange(n)] = 0.0
    return scale_to_load(topo, seq, load)


def scale_to_load(topo: Topology, seq: np.ndarray, load: float) -> np.ndarray:
    """Scale a TM sequence so its time-averaged TM, routed by hop-count shortest
    paths, loads the busiest link to `load` x capacity.

    load = 1.0 therefore means "default OSPF routing just saturates on average".
    """
    from sdnte.network import link_loads
    from sdnte.routing.baselines import shortest_path_routing

    routing = shortest_path_routing(topo)
    ref = (link_loads(topo, routing, seq.mean(axis=0)) / topo.capacities).max()
    return seq * (load / ref)
