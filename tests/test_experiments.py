import pandas as pd

from sdnte.config import load_config
from sdnte.experiments import run_all
from sdnte.plots import make_all


def test_grid_runner_and_plots(tmp_path):
    config = load_config()
    config["experiments"].update(topologies=["toy6"], methods=["ospf_hop", "ecmp", "greedy", "lp_optimal"], seeds=[0])
    config["traffic"].update(patterns=["gravity", "hotspot"], loads=[0.6, 0.9])
    results, figures = tmp_path / "results", tmp_path / "figures"

    episodes = run_all(config, quick=True, out_dir=results, log=lambda _: None)
    # 2 patterns x 2 loads x 4 methods normal runs + failure runs at loads 0.6 and 0.9
    assert len(episodes) == 2 * 2 * 4 * 2
    assert set(episodes["scenario"]) == {"normal", "failure"}
    failure = pd.read_csv(results / "failure_steps.csv")
    assert set(failure["step"]) == set(range(20))

    made = make_all(results, figures)
    assert all(p.exists() for p in made)
    summary = pd.read_csv(results / "summary_load0.9.csv")
    ospf = summary[summary["method"] == "ospf_hop"].iloc[0]
    assert ospf["mlu_reduction_vs_ospf_pct"] == 0
    assert (summary["gap_to_lp_pct"] >= -1e-6).all()
