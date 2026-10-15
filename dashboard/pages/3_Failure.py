import plotly.graph_objects as go
import streamlit as st

from common import color, label, load_results, no_results_message
from sdnte.plots import MAIN_METHODS, METHOD_STYLE, failure_effect

st.set_page_config(page_title="Failure", layout="wide")
st.title("Link failure: robustness")
st.markdown(
    "Each test sequence is replayed with a randomly chosen link (one whose loss keeps the network "
    "connected) going down part-way through. The controller is notified immediately and every method "
    "recomputes routes on the remaining links."
)

steps = load_results("failure_steps.csv")
if steps is None or "scenario" not in steps:
    no_results_message()
    st.stop()
effect = failure_effect(steps)

present = [m for m in METHOD_STYLE if m in set(effect["method"])]
c1, c2, c3 = st.columns(3)
topo = c1.selectbox("Topology", sorted(effect["topology"].unique()))
load = c2.selectbox("Load", sorted(effect["load"].unique()), index=len(effect["load"].unique()) - 1)
pattern = c3.selectbox("Traffic pattern", ["all"] + sorted(effect["pattern"].unique()))
methods = st.multiselect("Methods", present, default=[m for m in MAIN_METHODS if m in present], format_func=label)
view = st.radio("Show", ["extra_mlu", "mlu"], horizontal=True,
                format_func={"extra_mlu": "Extra MLU caused by the failure", "mlu": "MLU in the failure run"}.get)

df = effect[(effect["topology"] == topo) & (effect["load"] == load) & effect["method"].isin(methods)]
if pattern != "all":
    df = df[df["pattern"] == pattern]
if df.empty:
    st.info("No runs for this selection.")
    st.stop()
fail_step = int(df["fail_step"].iloc[0])

fig = go.Figure()
for m in [m for m in present if m in methods]:
    d = df[df["method"] == m].groupby("step")[view].mean()
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name=label(m), mode="lines",
                             line=dict(color=color(m), width=2)))
fig.add_vline(x=fail_step, line=dict(color="#d03b3b", width=1, dash="dash"),
              annotation_text="link failure", annotation_position="top")
y_title = {"extra_mlu": "Extra MLU vs. same traffic without failure", "mlu": "Max link utilization"}[view]
fig.update_layout(height=420, xaxis_title="Control interval", yaxis_title=y_title + " (mean over runs)",
                  hovermode="x unified", legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

after = df[df["step"] >= fail_step].groupby("method")
table = after[["mlu_no_failure", "mlu", "extra_mlu", "loss_pct_no_failure", "loss_pct"]].mean()
table.index = table.index.map(label)
st.subheader("After the failure")
st.dataframe(table.round(3).rename(columns={
    "mlu_no_failure": "MLU without failure", "mlu": "MLU with failure", "extra_mlu": "Extra MLU",
    "loss_pct_no_failure": "Loss % without failure", "loss_pct": "Loss % with failure"}), use_container_width=True)
st.caption(f"Averaged over the intervals after the failure. Failed links in these runs: "
           f"{', '.join(sorted(df['failed_link'].unique()))}")
