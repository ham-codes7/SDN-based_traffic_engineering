"""Logically centralized SDN controller (simulated).

Mirrors the structure of a real controller such as Ryu:
- southbound: installs flow rules into switch flow tables, and collects port stats
- northbound: exposes the network state to a pluggable traffic-engineering (TE) app

Each call to `step` is one control interval: the TE app decides a routing for the
measured traffic matrix, the controller installs it, the data plane carries the
traffic (`network.evaluate`), and port stats are collected for the next interval.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sdnte.network import Metrics, Routing, evaluate
from sdnte.topology import Topology


@dataclass
class NetworkState:
    """Northbound view of the network handed to the TE app each interval."""

    topology: Topology
    traffic_matrix: np.ndarray  # demand measured for this interval, Mbps (N x N)
    link_util: np.ndarray  # per-link utilization from the previous interval's port stats
    step: int


class TEApp(Protocol):
    """A traffic-engineering application: OSPF, ECMP, LP, or a DRL agent."""

    name: str

    def decide(self, state: NetworkState) -> Routing: ...


@dataclass
class FlowRule:
    match: tuple[int, int]  # (src switch, dst switch) of the traffic aggregate
    # (next-hop switch, share of traffic). More than one entry is an OpenFlow SELECT group.
    actions: list[tuple[int, float]]


@dataclass
class Switch:
    dpid: int
    flow_table: dict[tuple[int, int], FlowRule] = field(default_factory=dict)
    tx_mbps: dict[int, float] = field(default_factory=dict)  # per egress neighbour


class SDNController:
    def __init__(self, topology: Topology, te_app: TEApp):
        self.topology = topology.copy()
        self.te_app = te_app
        self.switches = {n: Switch(n) for n in range(topology.num_nodes)}
        self.link_util = np.zeros(topology.num_links)
        self.step_count = 0

    def fail_link(self, u: int, v: int) -> None:
        """Simulate a link-down event (as reported by an OpenFlow PORT_STATUS message)."""
        self.topology.fail_link(u, v)

    def install_routing(self, routing: Routing) -> None:
        """Translate path-based routing into per-switch flow rules."""
        for sw in self.switches.values():
            sw.flow_table.clear()
        for (s, d), splits in routing.items():
            next_hops: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
            for path, frac in splits:
                for a, b in zip(path, path[1:]):
                    next_hops[a][b] += frac
            for sw, hops in next_hops.items():
                total = sum(hops.values())
                actions = [(nh, w / total) for nh, w in hops.items()]
                self.switches[sw].flow_table[(s, d)] = FlowRule((s, d), actions)

    def collect_port_stats(self, metrics: Metrics) -> None:
        """Update switch port counters and the controller's link-utilization view."""
        for i, link in enumerate(self.topology.links):
            self.switches[link.src].tx_mbps[link.dst] = metrics.link_util[i] * link.capacity
        self.link_util = metrics.link_util.copy()

    @property
    def num_rules(self) -> int:
        return sum(len(sw.flow_table) for sw in self.switches.values())

    def step(self, tm: np.ndarray) -> tuple[Metrics, float]:
        """Run one control interval. Returns the data-plane metrics and the TE app's
        decision time in ms."""
        state = NetworkState(self.topology, tm, self.link_util.copy(), self.step_count)
        t0 = time.perf_counter()
        routing = self.te_app.decide(state)
        decision_ms = (time.perf_counter() - t0) * 1e3
        self.install_routing(routing)
        metrics = evaluate(self.topology, routing, tm)
        self.collect_port_stats(metrics)
        self.step_count += 1
        return metrics, decision_ms
