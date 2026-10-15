"""Shared helpers for the dashboard pages: cached simulation runs and topology drawing."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

from sdnte.controller import SDNController
from sdnte.experiments import RESULTS_DIR
from sdnte.plots import METHOD_STYLE
from sdnte.routing import available_methods, make_te_app
from sdnte.topology import Topology
from sdnte.traffic import generate_sequence

# Sequential blue ramp for utilization 0-100%; overloaded links (>100%) use the
# reserved "critical" status colour and are also labelled in the hover text.
UTIL_SCALE = [[0.0, "#cde2fb"], [0.35, "#86b6ef"], [0.6, "#3987e5"], [0.8, "#256abf"], [1.0, "#0d366b"]]
CRITICAL = "#d03b3b"
FAILED = "#9b9a94"


def label(method: str) -> str:
    return METHOD_STYLE.get(method, (method, None))[0]


def color(method: str) -> str:
    return METHOD_STYLE.get(method, (None, "#52514e"))[1]


@st.cache_resource
def load_topology(name: str) -> Topology:
    return Topology.load(name)


@st.cache_data
def methods_for(topology: str) -> list[str]:
    return available_methods(topology)


@st.cache_resource
def te_app(method: str, topology: str):
    return make_te_app(method, topology)


@st.cache_data(show_spinner="Running the SDN controller...")
def simulate(topology: str, method: str, pattern: str, load: float, seed: int, steps: int,
             fail_step: int | None, fail_link: tuple[int, int] | None) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-step metrics and per-step link utilization (steps x links) for one episode."""
    topo = load_topology(topology)
    tm_seq = generate_sequence(topo, pattern, steps, load, seed)
    controller = SDNController(topo, te_app(method, topology))
    rows, utils = [], []
    for t, tm in enumerate(tm_seq):
        if fail_step is not None and fail_link is not None and t == fail_step:
            controller.fail_link(*fail_link)
        metrics, decision_ms = controller.step(tm)
        rows.append({"step": t, **metrics.summary(), "decision_ms": decision_ms,
                     "flow_rules": controller.num_rules})
        utils.append(metrics.link_util)
    return pd.DataFrame(rows), np.array(utils)


def layout(topo: Topology) -> dict[int, tuple[float, float]]:
    if topo.positions:
        return topo.positions
    return nx.kamada_kawai_layout(nx.Graph(topo.graph().to_undirected()))


def topology_figure(topo: Topology, link_util: np.ndarray, up: np.ndarray, title: str) -> go.Figure:
    """Network map: each undirected link coloured by the busier of its two directions."""
    pos = layout(topo)
    fig = go.Figure()
    mid_x, mid_y, mid_text, mid_color = [], [], [], []
    max_cap = topo.capacities.max()
    for i, l in enumerate(topo.links):
        if l.src > l.dst:
            continue
        j = topo.link_index[(l.dst, l.src)]
        util = max(link_util[i], link_util[j])
        (x0, y0), (x1, y1) = pos[l.src], pos[l.dst]
        name = f"{topo.node_names[l.src]} – {topo.node_names[l.dst]}"
        if not up[i]:
            line, status = dict(color=FAILED, width=2, dash="dot"), "FAILED"
        elif util > 1.0:
            line, status = dict(color=CRITICAL, width=2 + 4 * l.capacity / max_cap, dash="dash"), "OVERLOADED"
        else:
            shade = sample_colorscale(UTIL_SCALE, [util])[0]
            line, status = dict(color=shade, width=2 + 4 * l.capacity / max_cap), ""
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=line, hoverinfo="skip", showlegend=False))
        mid_x.append((x0 + x1) / 2)
        mid_y.append((y0 + y1) / 2)
        mid_color.append(min(util, 1.0))
        mid_text.append(f"<b>{name}</b><br>utilization {util:.0%} {status}<br>"
                        f"→ {link_util[i]:.0%}  ← {link_util[j]:.0%}<br>capacity {l.capacity:g} Mbps")

    # Hover targets at link midpoints; this trace also carries the colour bar.
    fig.add_trace(go.Scatter(
        x=mid_x, y=mid_y, mode="markers", hovertext=mid_text, hoverinfo="text", showlegend=False,
        marker=dict(size=14, opacity=0.01, color=mid_color, colorscale=UTIL_SCALE, cmin=0, cmax=1,
                    colorbar=dict(title="Link util.", tickformat=".0%", thickness=12, len=0.8))))
    nodes = list(range(topo.num_nodes))
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
        text=topo.node_names, textposition="top center", hoverinfo="text", showlegend=False,
        marker=dict(size=14, color="#ffffff", line=dict(color="#52514e", width=2))))
    fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(bgcolor="white"))
    return fig


@st.cache_data
def load_results(name: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / name
    return pd.read_csv(path) if path.exists() else None


def no_results_message() -> None:
    st.warning("No experiment results yet. Run `uv run sdnte run-all` (or `--quick`) and reload.")

