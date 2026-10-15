"""Network topology: switches and links, loaded from JSON files in data/topologies/."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = REPO_ROOT / "data" / "topologies"


@dataclass(frozen=True)
class Link:
    src: int
    dst: int
    capacity: float  # Mbps
    delay: float  # propagation delay, ms


class Topology:
    """Directed view of an undirected topology.

    Each undirected link in the JSON file becomes two directed links. Link indices
    never change, even when a link fails, so per-link vectors (utilization, DRL link
    weights) keep a fixed layout. Failed links are tracked in the `up` mask.
    """

    def __init__(self, name: str, node_names: list[str], links: list[Link]):
        self.name = name
        self.node_names = list(node_names)
        self.links = list(links)
        self.link_index = {(l.src, l.dst): i for i, l in enumerate(self.links)}
        self.capacities = np.array([l.capacity for l in self.links], dtype=float)
        self.delays = np.array([l.delay for l in self.links], dtype=float)
        self.up = np.ones(len(self.links), dtype=bool)

    @classmethod
    def load(cls, name_or_path: str | Path) -> Topology:
        """Load a topology by name (e.g. "abilene") or by path to a JSON file."""
        path = Path(name_or_path)
        if path.suffix != ".json":
            path = TOPOLOGY_DIR / f"{name_or_path}.json"
        data = json.loads(path.read_text())
        names = data["nodes"]
        idx = {n: i for i, n in enumerate(names)}
        links = []
        for e in data["links"]:
            u, v = idx[e["src"]], idx[e["dst"]]
            for a, b in ((u, v), (v, u)):
                links.append(Link(a, b, float(e["capacity"]), float(e.get("delay", 1.0))))
        return cls(data["name"], names, links)

    @staticmethod
    def available() -> list[str]:
        return sorted(p.stem for p in TOPOLOGY_DIR.glob("*.json"))

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    @property
    def num_links(self) -> int:
        return len(self.links)

    @property
    def od_pairs(self) -> list[tuple[int, int]]:
        n = self.num_nodes
        return [(s, d) for s in range(n) for d in range(n) if s != d]

    def graph(self) -> nx.DiGraph:
        """Directed graph of the links that are currently up."""
        g = nx.DiGraph()
        g.add_nodes_from(range(self.num_nodes))
        for i, l in enumerate(self.links):
            if self.up[i]:
                g.add_edge(l.src, l.dst, index=i, capacity=l.capacity, delay=l.delay)
        return g

    def path_links(self, path: list[int]) -> list[int]:
        """Link indices traversed by a node path."""
        return [self.link_index[(a, b)] for a, b in zip(path, path[1:])]

    def fail_link(self, u: int, v: int) -> None:
        """Take down the link between u and v in both directions."""
        self.up[self.link_index[(u, v)]] = False
        self.up[self.link_index[(v, u)]] = False

    def restore_all(self) -> None:
        self.up[:] = True

    def copy(self) -> Topology:
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"Topology({self.name!r}, nodes={self.num_nodes}, directed_links={self.num_links})"
