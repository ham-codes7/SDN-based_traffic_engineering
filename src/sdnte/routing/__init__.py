"""TE applications, looked up by name from the CLI and experiment runner."""

from __future__ import annotations

from sdnte.controller import TEApp
from sdnte.routing.baselines import ECMP, GreedyLeastLoaded, OSPFHop, OSPFInvCap
from sdnte.routing.optimal import LPOptimal

BASELINES = {cls.name: cls for cls in (OSPFHop, OSPFInvCap, ECMP, GreedyLeastLoaded, LPOptimal)}


def make_te_app(name: str) -> TEApp:
    """Create a TE app by name. DRL agents ("ppo", "a2c") are added in agents.py."""
    if name not in BASELINES:
        raise ValueError(f"Unknown method {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name]()
