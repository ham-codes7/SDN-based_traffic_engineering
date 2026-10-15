"""Streamlit dashboard (owner P4, weeks 4-5). Run: uv run streamlit run dashboard/app.py

Planned pages:
- Simulate: pick topology / pattern / load / method, step through time on a Plotly
  topology with links coloured by utilization; OSPF vs DRL side by side
- Results: comparison charts from results/*.csv
- Training: PPO / A2C reward curves
- Failure: MLU timeline around a link failure
"""

import streamlit as st

st.set_page_config(page_title="SDN DRL Traffic Engineering", layout="wide")
st.title("SDN-Based Intelligent Traffic Engineering")
st.info("Dashboard not built yet. Try `uv run sdnte simulate --topo toy6` from the terminal.")
