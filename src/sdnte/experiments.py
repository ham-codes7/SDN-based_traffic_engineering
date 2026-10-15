"""Experiment runner: drives the SDN controller over traffic-matrix sequences."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from sdnte.controller import SDNController, TEApp
from sdnte.topology import REPO_ROOT, Topology

RESULTS_DIR = REPO_ROOT / "results"
METRICS = ["mlu", "mean_util", "loss_pct", "throughput_ratio", "avg_delay_ms", "delay_std_ms", "decision_ms"]


def run_episode(
    topology: Topology,
    te_app: TEApp,
    tm_seq: np.ndarray,
    fail_at: int | None = None,
    fail_link: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Run one episode (one TM per control interval) and return per-step metrics.

    If `fail_at` and `fail_link` are given, that link goes down at step `fail_at`.
    """
    controller = SDNController(topology, te_app)
    rows = []
    for t, tm in enumerate(tm_seq):
        if fail_at is not None and fail_link is not None and t == fail_at:
            controller.fail_link(*fail_link)
        metrics, decision_ms = controller.step(tm)
        rows.append(
            {
                "step": t,
                "method": te_app.name,
                **metrics.summary(),
                "decision_ms": decision_ms,
                "flow_rules": controller.num_rules,
            }
        )
    return pd.DataFrame(rows)


def run_all(
    config: dict,
    quick: bool = False,
    out_dir: Path = RESULTS_DIR,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Full experiment grid from configs/default.yaml.

    For every topology x traffic pattern x load x seed, every method routes the same
    TM sequence. At the failure loads, each sequence is replayed with a random
    (non-partitioning) link failing part-way through.

    Writes:
      results/episodes.csv       one row per (condition, method, scenario): episode means
      results/failure_steps.csv  per-step metrics for the failure runs
    `quick` uses 2 seeds and 20-step episodes, for checking the pipeline.
    """
    from sdnte.routing import available_methods, make_te_app
    from sdnte.traffic import generate_sequence

    exp = config["experiments"]
    steps = 20 if quick else config["episode_steps"]
    seeds = exp["seeds"][:2] if quick else exp["seeds"]
    fail_step = min(exp["failure"]["step"], steps // 2)
    offset = exp["test_seed_offset"]

    episodes, failure_steps = [], []
    for topo_name in exp["topologies"]:
        topo = Topology.load(topo_name)
        available = available_methods(topo_name)
        methods = [m for m in exp["methods"] if m in available]
        missing = [m for m in exp["methods"] if m not in available]
        if missing:
            log(f"[{topo_name}] skipping untrained agents: {', '.join(missing)} (run `sdnte train`)")
        apps = {m: make_te_app(m, topo_name) for m in methods}
        failure_links = topo.safe_failure_links()

        for pattern in config["traffic"]["patterns"]:
            for load in config["traffic"]["loads"]:
                log(f"[{topo_name}] {pattern} load {load}")
                for seed in seeds:
                    tm_seq = generate_sequence(topo, pattern, steps, load, seed=offset + seed)
                    cond = {"topology": topo_name, "pattern": pattern, "load": load, "seed": seed}
                    for m, app in apps.items():
                        df = run_episode(topo, app, tm_seq)
                        episodes.append({**cond, "scenario": "normal", "method": m, **df[METRICS].mean()})

                    if load not in exp["failure"]["loads"]:
                        continue
                    rng = np.random.default_rng(offset + seed)
                    link = failure_links[rng.integers(len(failure_links))]
                    link_name = f"{topo.node_names[link[0]]}-{topo.node_names[link[1]]}"
                    for m, app in apps.items():
                        df = run_episode(topo, app, tm_seq, fail_at=fail_step, fail_link=link)
                        episodes.append({**cond, "scenario": "failure", "method": m, **df[METRICS].mean()})
                        failure_steps.append(df.assign(**cond, failed_link=link_name, fail_step=fail_step))

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_df = pd.DataFrame(episodes)
    episodes_df.to_csv(out_dir / "episodes.csv", index=False)
    if failure_steps:
        pd.concat(failure_steps, ignore_index=True).to_csv(out_dir / "failure_steps.csv", index=False)
    return episodes_df
