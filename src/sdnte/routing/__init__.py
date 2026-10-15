"""TE applications, looked up by name from the CLI and experiment runner."""

from __future__ import annotations

from sdnte.controller import TEApp
from sdnte.routing.baselines import ECMP, GreedyLeastLoaded, OSPFHop, OSPFInvCap
from sdnte.routing.optimal import LPOptimal

BASELINES = {cls.name: cls for cls in (OSPFHop, OSPFInvCap, ECMP, GreedyLeastLoaded, LPOptimal)}


def available_methods(topology: str) -> list[str]:
    """Baselines plus every DRL agent trained for this topology."""
    from sdnte.agents import trained_agents

    return list(BASELINES) + trained_agents(topology)


def make_te_app(name: str, topology: str | None = None) -> TEApp:
    """Create a TE app by name. DRL agents (e.g. "ppo") need the topology they were
    trained on, since the policy's input/output size depends on it."""
    if name in BASELINES:
        return BASELINES[name]()
    if topology is None:
        raise ValueError(f"Unknown method {name!r}; baselines: {sorted(BASELINES)}")
    from sdnte.agents import DRLTEApp

    return DRLTEApp(topology, name)
