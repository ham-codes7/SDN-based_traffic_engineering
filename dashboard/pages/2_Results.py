import plotly.graph_objects as go
import streamlit as st

from common import color, label, load_results, no_results_message
from sdnte.plots import MAIN_METHODS, METHOD_STYLE, summary_table

st.set_page_config(page_title="Results", layout="wide")
st.title("Results: experiment grid")

episodes = load_results("episodes.csv")
if episodes is None:
    no_results_message()
    st.stop()

METRIC_NAMES = {"mlu": "Max link utilization", "loss_pct": "Packet loss (%)",
                "avg_delay_ms": "Avg end-to-end delay (ms)", "throughput_ratio": "Throughput ratio"}
present = [m for m in METHOD_STYLE if m in set(episodes["method"])]

c1, c2, c3 = st.columns(3)
topo = c1.selectbox("Topology", sorted(episodes["topology"].unique()))
metric = c2.selectbox("Metric", list(METRIC_NAMES), format_func=METRIC_NAMES.get)
scenario = c3.selectbox("Scenario", ["normal", "failure"], format_func={"normal": "No failure", "failure": "Link failure"}.get)
methods = st.multiselect("Methods", present, default=[m for m in MAIN_METHODS if m in present], format_func=label)

df = episodes[(episodes["topology"] == topo) & (episodes["scenario"] == scenario) & episodes["method"].isin(methods)]
if df.empty:
    st.info("No episodes for this selection.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader(f"{METRIC_NAMES[metric]} vs load")
    agg = df.groupby(["method", "load"])[metric].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    for m in [m for m in present if m in methods]:
        d = agg[agg["method"] == m]
        fig.add_trace(go.Scatter(x=d["load"], y=d["mean"], name=label(m), mode="lines+markers",
                                 error_y=dict(type="data", array=d["std"], thickness=1, width=4),
                                 line=dict(color=color(m), width=2), marker=dict(size=8)))
    fig.update_layout(height=380, xaxis_title="Offered load (1.0 = OSPF saturation)", yaxis_title=METRIC_NAMES[metric],
                      hovermode="x unified", legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mean over traffic patterns and seeds; error bars = 1 standard deviation.")

with right:
    loads = sorted(df["load"].unique())
    load = st.select_slider("Load for the per-pattern view", loads, value=0.9 if 0.9 in loads else loads[-1])
    st.subheader(f"{METRIC_NAMES[metric]} by traffic pattern")
    d = df[df["load"] == load].groupby(["pattern", "method"])[metric].agg(["mean", "std"]).reset_index()
    fig = go.Figure()
    for m in [m for m in present if m in methods]:
        dm = d[d["method"] == m]
        fig.add_trace(go.Bar(x=dm["pattern"], y=dm["mean"], name=label(m), marker_color=color(m),
                             error_y=dict(type="data", array=dm["std"], thickness=1, width=3)))
    fig.update_layout(height=380, barmode="group", bargap=0.25, bargroupgap=0.05, yaxis_title=METRIC_NAMES[metric],
                      legend=dict(orientation="h", y=-0.25), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Summary at load {load}")
table = summary_table(episodes, load=load)
table = table[table["topology"] == topo].drop(columns="topology")
table["method"] = table["method"].map(label)
st.dataframe(
    table.rename(columns={"mlu": "MLU", "loss_pct": "Loss %", "avg_delay_ms": "Delay (ms)",
                          "decision_ms": "Decision time (ms)", "mlu_under_failure": "MLU with failure",
                          "mlu_reduction_vs_ospf_pct": "MLU reduction vs OSPF (%)",
                          "gap_to_lp_pct": "Gap to LP optimum (%)"}),
    hide_index=True, use_container_width=True)
st.caption("Averaged over traffic patterns and seeds (no-failure runs, except the 'MLU with failure' column).")

with st.expander("Raw episode results"):
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", df.to_csv(index=False), file_name=f"episodes_{topo}_{scenario}.csv")
