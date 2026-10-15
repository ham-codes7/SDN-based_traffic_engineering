"""Report figures and summary table from results/*.csv (`uv run sdnte plots`)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sdnte.experiments import RESULTS_DIR
from sdnte.topology import REPO_ROOT

FIGURES_DIR = REPO_ROOT / "figures"
MODELS_LOG_DIR = REPO_ROOT / "models" / "logs"

# Each method keeps the same colour in every figure (colour follows the entity).
METHOD_STYLE = {
    "ppo": ("DRL-PPO (ours)", "#2a78d6"),
    "ospf_hop": ("OSPF (hop count)", "#eb6834"),
    "ecmp": ("ECMP", "#1baf7a"),
    "greedy": ("Greedy least-loaded", "#eda100"),
    "a2c": ("DRL-A2C", "#e87ba4"),
    "ospf_invcap": ("OSPF (inv. capacity)", "#008300"),
    "ppo_gravity": ("PPO trained on gravity only", "#4a3aa7"),
    "lp_optimal": ("LP optimal (bound)", "#e34948"),
}
MAIN_METHODS = ["ospf_hop", "ecmp", "greedy", "ppo", "lp_optimal"]
METRIC_LABELS = {
    "mlu": "Max link utilization",
    "loss_pct": "Packet loss (%)",
    "avg_delay_ms": "Avg end-to-end delay (ms)",
    "throughput_ratio": "Throughput (delivered / offered)",
}


def _label(m: str) -> str:
    return METHOD_STYLE.get(m, (m, None))[0]


def _palette(methods) -> dict[str, str]:
    return {_label(m): METHOD_STYLE[m][1] for m in methods if m in METHOD_STYLE}


def _present(df: pd.DataFrame, methods: list[str]) -> list[str]:
    return [m for m in methods if m in set(df["method"])]


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"grid.color": "#e6e5e1", "axes.edgecolor": "#c3c2b7", "lines.linewidth": 2,
                         "legend.frameon": False, "savefig.bbox": "tight"})


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    return out_dir / f"{name}.png"


def _legend_below(fig, axes) -> None:
    """One legend for the whole figure, below the panels, so it never covers data."""
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0]:
        if ax.get_legend():
            ax.get_legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=min(len(labels), 5), fontsize=9)
    fig.tight_layout()


def _with_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Method=df["method"].map(_label))


def metric_vs_load(episodes: pd.DataFrame, metric: str, out_dir: Path) -> Path:
    """Metric vs load, one panel per topology, mean and 95% CI over patterns x seeds."""
    df = episodes[episodes["scenario"] == "normal"]
    methods = _present(df, MAIN_METHODS)
    df = _with_labels(df[df["method"].isin(methods)])
    topos = list(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topos), figsize=(5.2 * len(topos), 3.8), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        sns.lineplot(data=df[df["topology"] == topo], x="load", y=metric, hue="Method",
                     hue_order=[_label(m) for m in methods], palette=_palette(methods),
                     marker="o", markersize=7, errorbar=("ci", 95), ax=ax, legend=ax is axes[0][0])
        ax.set_title(topo.capitalize())
        ax.set_xlabel("Offered load (1.0 = OSPF saturation)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    _legend_below(fig, axes)
    return _save(fig, f"{metric}_vs_load", out_dir)


def mlu_by_pattern(episodes: pd.DataFrame, out_dir: Path, load: float = 0.9) -> Path:
    df = episodes[(episodes["scenario"] == "normal") & (episodes["load"] == load)]
    methods = _present(df, MAIN_METHODS)
    df = _with_labels(df[df["method"].isin(methods)])
    topos = list(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topos), figsize=(5.8 * len(topos), 3.8), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        sns.barplot(data=df[df["topology"] == topo], x="pattern", y="mlu", hue="Method",
                    hue_order=[_label(m) for m in methods], palette=_palette(methods),
                    errorbar=("ci", 95), err_kws={"linewidth": 1}, ax=ax, legend=ax is axes[0][0])
        ax.set_title(f"{topo.capitalize()}, load {load}")
        ax.set_xlabel("Traffic pattern")
        ax.set_ylabel(METRIC_LABELS["mlu"])
    _legend_below(fig, axes)
    return _save(fig, "mlu_by_pattern", out_dir)


def failure_effect(failure_steps: pd.DataFrame) -> pd.DataFrame:
    """Per-step metrics of each failure run next to the same run without the failure.

    Adds `extra_mlu` and `extra_loss_pct`: how much the failure itself raised MLU and
    loss for that method on identical traffic (removes the traffic's own variation).
    """
    keys = ["topology", "pattern", "load", "seed", "method", "step"]
    cols = keys + ["mlu", "loss_pct"]
    normal = failure_steps[failure_steps["scenario"] == "normal"][cols]
    failed = failure_steps[failure_steps["scenario"] == "failure"]
    df = failed.merge(normal, on=keys, suffixes=("", "_no_failure"))
    df["extra_mlu"] = df["mlu"] - df["mlu_no_failure"]
    df["extra_loss_pct"] = df["loss_pct"] - df["loss_pct_no_failure"]
    return df


def failure_timeline(failure_steps: pd.DataFrame, out_dir: Path, load: float = 0.9) -> Path:
    df = failure_effect(failure_steps)
    df = df[df["load"] == load]
    methods = _present(df, MAIN_METHODS)
    df = _with_labels(df[df["method"].isin(methods)])
    topos = list(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topos), figsize=(5.2 * len(topos), 3.8), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        d = df[df["topology"] == topo]
        sns.lineplot(data=d, x="step", y="extra_mlu", hue="Method", hue_order=[_label(m) for m in methods],
                     palette=_palette(methods), errorbar=None, ax=ax, legend=ax is axes[0][0])
        ax.axvline(d["fail_step"].iloc[0], color="#52514e", linestyle="--", linewidth=1)
        ax.text(d["fail_step"].iloc[0] + 0.5, ax.get_ylim()[1] * 0.97, "link failure", va="top",
                fontsize=8, color="#52514e")
        ax.set_title(f"{topo.capitalize()}, load {load}")
        ax.set_xlabel("Control interval")
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_ylabel("Extra max link utilization\ncaused by the failure")
    _legend_below(fig, axes)
    return _save(fig, "failure_timeline", out_dir)


def generalization(episodes: pd.DataFrame, out_dir: Path, load: float = 0.9) -> Path | None:
    methods = ["ospf_hop", "ppo_gravity", "ppo", "lp_optimal"]
    df = episodes[(episodes["scenario"] == "normal") & (episodes["load"] == load)]
    if "ppo_gravity" not in set(df["method"]):
        return None
    methods = _present(df, methods)
    df = _with_labels(df[df["method"].isin(methods)])
    topos = list(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topos), figsize=(5.8 * len(topos), 3.8), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        sns.barplot(data=df[df["topology"] == topo], x="pattern", y="mlu", hue="Method",
                    hue_order=[_label(m) for m in methods], palette=_palette(methods),
                    errorbar=("ci", 95), err_kws={"linewidth": 1}, ax=ax, legend=ax is axes[0][0])
        ax.set_title(f"{topo.capitalize()}: generalization to unseen patterns")
        ax.set_xlabel("Test traffic pattern (gravity = seen in training)")
        ax.set_ylabel(METRIC_LABELS["mlu"])
    _legend_below(fig, axes)
    return _save(fig, "generalization", out_dir)


def load_training_logs(log_root: Path = MODELS_LOG_DIR) -> pd.DataFrame:
    """Episode rewards from the SB3 Monitor CSVs, with cumulative timesteps."""
    frames = []
    for run_dir in sorted(p for p in log_root.glob("*") if p.is_dir()):
        topo, _, agent = run_dir.name.partition("_")
        parts = [pd.read_csv(f, skiprows=1) for f in sorted(run_dir.glob("*.monitor.csv"))]
        if not parts:
            continue
        df = pd.concat(parts).sort_values("t").reset_index(drop=True)
        df["timesteps"] = df["l"].cumsum()
        df["reward_smooth"] = df["r"].rolling(50, min_periods=1).mean()
        frames.append(df.assign(topology=topo, method=agent))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def training_curves(out_dir: Path, topologies: list[str], log_root: Path = MODELS_LOG_DIR) -> Path | None:
    df = load_training_logs(log_root)
    if df.empty:
        return None
    df = df[df["topology"].isin(topologies)]
    if df.empty:
        return None
    methods = _present(df, ["ppo", "a2c", "ppo_gravity"])
    df = _with_labels(df)
    topos = list(df["topology"].unique())
    fig, axes = plt.subplots(1, len(topos), figsize=(5.2 * len(topos), 3.6), sharey=True, squeeze=False)
    for ax, topo in zip(axes[0], topos):
        sns.lineplot(data=df[df["topology"] == topo], x="timesteps", y="reward_smooth", hue="Method",
                     hue_order=[_label(m) for m in methods], palette=_palette(methods), errorbar=None,
                     ax=ax, legend=ax is axes[0][0])
        ax.axhline(0, color="#52514e", linestyle="--", linewidth=1)
        ax.set_title(f"{topo.capitalize()} training")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Episode reward (higher is better)")
    _legend_below(fig, axes)
    return _save(fig, "training_curves", out_dir)


def summary_table(episodes: pd.DataFrame, load: float = 0.9) -> pd.DataFrame:
    """Per topology and method at one load, averaged over patterns and seeds:
    MLU, loss, delay, decision time, improvement vs OSPF-hop and gap to LP optimum."""
    df = episodes[episodes["load"] == load]
    normal = df[df["scenario"] == "normal"].groupby(["topology", "method"])
    table = normal[["mlu", "loss_pct", "avg_delay_ms", "decision_ms"]].mean()
    table["mlu_under_failure"] = df[df["scenario"] == "failure"].groupby(["topology", "method"])["mlu"].mean()
    ospf = table.xs("ospf_hop", level="method")["mlu"]
    lp = table.xs("lp_optimal", level="method")["mlu"] if "lp_optimal" in set(df["method"]) else None
    topo_idx = table.index.get_level_values("topology")
    table["mlu_reduction_vs_ospf_pct"] = 100 * (1 - table["mlu"].values / ospf.loc[topo_idx].values)
    if lp is not None:
        table["gap_to_lp_pct"] = 100 * (table["mlu"].values / lp.loc[topo_idx].values - 1)
    order = {m: i for i, m in enumerate(METHOD_STYLE)}
    table = table.reset_index().sort_values(["topology", "method"], key=lambda s: s.map(order) if s.name == "method" else s)
    return table.round(3)


def make_all(results_dir: Path = RESULTS_DIR, out_dir: Path = FIGURES_DIR) -> list[Path]:
    _style()
    episodes = pd.read_csv(results_dir / "episodes.csv")
    made = [metric_vs_load(episodes, m, out_dir) for m in ("mlu", "loss_pct", "avg_delay_ms")]
    made.append(mlu_by_pattern(episodes, out_dir))
    if (results_dir / "failure_steps.csv").exists():
        made.append(failure_timeline(pd.read_csv(results_dir / "failure_steps.csv"), out_dir))
    made += [p for p in (generalization(episodes, out_dir), training_curves(out_dir, list(episodes["topology"].unique()))) if p]

    table = summary_table(episodes)
    table.to_csv(results_dir / "summary_load0.9.csv", index=False)
    (results_dir / "summary_load0.9.md").write_text(table.to_markdown(index=False))
    return made
