import json

import plotly.graph_objects as go
import streamlit as st

from common import color, label
from sdnte.agents import MODELS_DIR
from sdnte.plots import load_training_logs

st.set_page_config(page_title="Training", layout="wide")
st.title("Training: DRL agents")
st.markdown(
    "Reward per step = relative max-link-utilization gain over hop-count OSPF on the same traffic, "
    "minus a packet-loss penalty. Curves show the stochastic training policy (smoothed over 50 episodes); "
    "the deployed policy acts deterministically and scores higher."
)

logs = load_training_logs()
if logs.empty:
    st.warning("No training logs yet. Run `uv run sdnte train`.")
    st.stop()

topo = st.selectbox("Topology", sorted(logs["topology"].unique()))
d = logs[logs["topology"] == topo]
fig = go.Figure()
for m in sorted(d["method"].unique()):
    dm = d[d["method"] == m]
    fig.add_trace(go.Scatter(x=dm["timesteps"], y=dm["reward_smooth"], name=label(m), mode="lines",
                             line=dict(color=color(m), width=2)))
fig.add_hline(y=0, line=dict(color="#52514e", width=1, dash="dash"))
fig.update_layout(height=420, xaxis_title="Environment steps", yaxis_title="Episode reward (higher is better)",
                  hovermode="x unified", legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

rows = []
for meta_path in sorted(MODELS_DIR.glob(f"{topo}_*.json")):
    meta = json.loads(meta_path.read_text())
    env = meta["env_kwargs"]
    rows.append({"agent": label(meta_path.stem.removeprefix(f"{topo}_")), "algorithm": meta["algo"].upper(),
                 "timesteps": meta["timesteps"], "training time (s)": meta["train_seconds"],
                 "traffic patterns": ", ".join(env["patterns"]), "load range": str(env["load_range"]),
                 "failure prob.": env["failure_prob"]})
if rows:
    st.subheader("Trained models")
    st.dataframe(rows, hide_index=True, use_container_width=True)
