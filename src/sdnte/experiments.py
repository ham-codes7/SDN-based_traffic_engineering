"""Experiment runner: drives the SDN controller over a traffic-matrix sequence."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sdnte.controller import SDNController, TEApp
from sdnte.topology import Topology


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


def run_all(config: dict) -> pd.DataFrame:
    """Full experiment grid from configs/default.yaml
    (topologies x methods x patterns x loads x seeds, plus failure runs).

    TODO (P3, week 4): loop over the grid, call run_episode, tag rows with the
    condition columns, and write results/results.csv.
    """
    raise NotImplementedError("Experiment grid runner is not implemented yet (week 4, owner P3)")
