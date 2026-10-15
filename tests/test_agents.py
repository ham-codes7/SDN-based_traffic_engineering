import pytest

from sdnte import agents
from sdnte.config import load_config
from sdnte.experiments import run_episode
from sdnte.routing import available_methods, make_te_app
from sdnte.topology import Topology
from sdnte.traffic import generate_sequence


@pytest.fixture
def tmp_models(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "MODELS_DIR", tmp_path)
    return tmp_path


def test_train_and_deploy_agent_in_controller(tmp_models):
    config = load_config()
    config["training"]["n_envs"] = 1
    path = agents.train("toy6", "ppo", config, timesteps=600)
    assert path.exists() and path.with_suffix(".json").exists()
    assert "ppo" in available_methods("toy6")

    topo = Topology.load("toy6")
    tm_seq = generate_sequence(topo, "hotspot", 6, 0.9, seed=0)
    df = run_episode(topo, make_te_app("ppo", "toy6"), tm_seq, fail_at=3, fail_link=(0, 3))
    assert len(df) == 6
    assert (df["loss_pct"] >= 0).all() and (df["mlu"] > 0).all()


def test_missing_model_gives_clear_error(tmp_models):
    with pytest.raises(FileNotFoundError, match="sdnte train"):
        make_te_app("ppo", "abilene")
