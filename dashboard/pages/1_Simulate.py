import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import color, label, load_topology, methods_for, simulate, topology_figure
from sdnte.topology import Topology
from sdnte.traffic import PATTERNS

st.set_page_config(page_title="Simulate", layout="wide")
st.title("Simulate: side-by-side comparison")

with st.sidebar:
    st.header("Scenario")
    topo_name = st.selectbox("Topology", Topology.available(), index=Topology.available().index("abilene"))
    topo = load_topology(topo_name)
    pattern = st.selectbox("Traffic pattern", PATTERNS, index=PATTERNS.index("hotspot"))
    load = st.slider("Offered load (1.0 = OSPF saturation)", 0.3, 1.2, 0.9, 0.05)
    steps = st.slider("Control intervals", 10, 60, 30, 5)
    seed = st.number_input("Traffic seed", 0, 99999, 1)
    failure = st.checkbox("Fail a link during the run")
    fail_step, fail_link = None, None
    if failure:
        candidates = topo.safe_failure_links()
        names = [f"{topo.node_names[u]} – {topo.node_names[v]}" for u, v in candidates]
        fail_link = candidates[names.index(st.selectbox("Link", names))]
        fail_step = st.slider("Failure at interval", 1, steps - 1, steps // 2)

    st.header("Methods")
    methods = methods_for(topo_name)
    left = st.selectbox("Left", methods, index=methods.index("ospf_hop"), format_func=label)
    right_default = "ppo" if "ppo" in methods else "greedy"
    right = st.selectbox("Right", methods, index=methods.index(right_default), format_func=label)

runs = {m: simulate(topo_name, m, pattern, load, int(seed), steps, fail_step, fail_link) for m in (left, right)}

t = st.slider("Control interval", 0, steps - 1, fail_step if failure else 0)
up = np.ones(topo.num_links, dtype=bool)
if failure and t >= fail_step:
    up[[topo.link_index[fail_link], topo.link_index[fail_link[::-1]]]] = False

cols = st.columns(2)
for col, m in zip(cols, (left, right)):
    df, utils = runs[m]
    row = df.iloc[t]
    other = runs[right if m == left else left][0].iloc[t]
    with col:
        st.plotly_chart(topology_figure(topo, utils[t], up, label(m)), use_container_width=True)
        a, b = st.columns(2)
        c, d = st.columns(2)
        a.metric("Max link util.", f"{row.mlu:.0%}", f"{(row.mlu - other.mlu) * 100:+.1f} pts", delta_color="inverse")
        b.metric("Packet loss", f"{row.loss_pct:.2f}%", f"{row.loss_pct - other.loss_pct:+.2f}", delta_color="inverse")
        c.metric("Avg delay", f"{row.avg_delay_ms:.1f} ms", f"{row.avg_delay_ms - other.avg_delay_ms:+.1f}", delta_color="inverse")
        d.metric("Flow rules", int(row.flow_rules))
st.caption("Deltas compare each method with the other one at the same interval. "
           "Links are coloured by utilization; dashed red = overloaded (>100%), dotted grey = failed.")

st.subheader("Over the whole run")
metric = st.radio("Metric", ["mlu", "loss_pct", "avg_delay_ms", "throughput_ratio"], horizontal=True,
                  format_func={"mlu": "Max link utilization", "loss_pct": "Packet loss (%)",
                               "avg_delay_ms": "Avg delay (ms)", "throughput_ratio": "Throughput ratio"}.get)
fig = go.Figure()
for m in dict.fromkeys((left, right)):
    df = runs[m][0]
    fig.add_trace(go.Scatter(x=df.step, y=df[metric], name=label(m), mode="lines+markers",
                             line=dict(color=color(m), width=2), marker=dict(size=8)))
fig.add_vline(x=t, line=dict(color="#52514e", width=1))
if failure:
    fig.add_vline(x=fail_step, line=dict(color="#d03b3b", width=1, dash="dash"),
                  annotation_text="link failure", annotation_position="top")
fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  xaxis_title="Control interval", legend=dict(orientation="h", y=-0.25))
st.plotly_chart(fig, use_container_width=True)

summary = {label(m): runs[m][0][["mlu", "loss_pct", "avg_delay_ms", "throughput_ratio", "decision_ms"]].mean()
           for m in dict.fromkeys((left, right))}
st.dataframe(pd.DataFrame(summary).T.round(3), use_container_width=True)
