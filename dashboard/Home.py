"""Streamlit dashboard. Run: uv run streamlit run dashboard/Home.py"""

import streamlit as st

from common import load_results, methods_for

st.set_page_config(page_title="SDN DRL Traffic Engineering", layout="wide")
st.title("SDN-Based Intelligent Traffic Engineering with Deep RL")

st.markdown(
    """
A centralized **SDN controller** collects port statistics from every switch and hands the
network state to a **traffic-engineering (TE) app**. The app decides how each
source–destination pair is routed, and the controller installs the matching flow rules.

Our TE app is a **deep reinforcement learning agent (PPO)**. Each control interval it observes
link utilization, link status and the measured traffic matrix, and outputs a weight for every
link. Traffic then follows the weighted shortest paths. It is compared with:

| Method | Type |
|---|---|
| OSPF (hop count), OSPF (inverse capacity) | Static shortest-path routing, as in conventional networks |
| ECMP | Equal split over all minimum-hop paths |
| Greedy least-loaded | Adaptive heuristic using the same traffic information as the agent |
| DRL-A2C | Second learning baseline |
| LP optimal | Best achievable max link utilization (upper bound, too slow for real time at scale) |
"""
)

st.subheader("Pages")
st.markdown(
    """
- **Simulate**: run any two methods side by side on the same traffic and watch link utilization step by step.
- **Results**: comparison across topologies, loads and traffic patterns from the experiment grid.
- **Failure**: how each method copes when a link goes down mid-run.
- **Training**: learning curves of the DRL agents.
"""
)

col1, col2 = st.columns(2)
for col, topo in zip((col1, col2), ("abilene", "nsfnet")):
    agents = [m for m in methods_for(topo) if m in ("ppo", "a2c", "ppo_gravity")]
    col.metric(f"Trained agents ({topo})", len(agents), help=", ".join(agents) or "none: run `uv run sdnte train`")
results = load_results("episodes.csv")
st.caption(
    f"Experiment results: {len(results)} episodes loaded." if results is not None
    else "Experiment results: none yet (run `uv run sdnte run-all`)."
)
