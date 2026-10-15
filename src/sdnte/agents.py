"""DRL agents: training with Stable-Baselines3, and the DRL TE app used at run time."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from sdnte.controller import NetworkState
from sdnte.env import TEEnv, action_to_weights, build_observation
from sdnte.network import Routing
from sdnte.routing.baselines import shortest_path_routing
from sdnte.topology import REPO_ROOT

MODELS_DIR = REPO_ROOT / "models"
ALGOS = {"ppo": PPO, "a2c": A2C}
HYPERPARAMS = {
    # gamma is low because a routing decision mostly affects the current interval.
    # log_std_init=-1 starts exploration near equal weights (= OSPF) instead of random routing.
    "ppo": dict(n_steps=512, batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.9,
                policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0)),
    "a2c": dict(n_steps=8, learning_rate=7e-4, gamma=0.9, policy_kwargs=dict(net_arch=[128, 128], log_std_init=-1.0)),
}


def model_path(topology: str, name: str) -> Path:
    return MODELS_DIR / f"{topology}_{name}.zip"


def log_dir(topology: str, name: str) -> Path:
    return MODELS_DIR / "logs" / f"{topology}_{name}"


def trained_agents(topology: str) -> list[str]:
    """Names of agents with a saved model for this topology."""
    return sorted(p.stem.removeprefix(f"{topology}_") for p in MODELS_DIR.glob(f"{topology}_*.zip"))


def train(topology: str, name: str, config: dict, timesteps: int | None = None, seed: int = 0) -> Path:
    """Train agent `name` (from config["training"]["agents"]) on `topology`."""
    tcfg = config["training"]
    spec = tcfg["agents"][name]
    env_kwargs = dict(
        topology=topology,
        patterns=tuple(spec["patterns"]),
        load_range=tuple(tcfg["load_range"]),
        episode_steps=config["episode_steps"],
        weight_range=tuple(tcfg["weight_range"]),
        loss_weight=tcfg["loss_weight"],
        failure_prob=tcfg["failure_prob"],
    )
    logs = log_dir(topology, name)
    logs.mkdir(parents=True, exist_ok=True)
    for old in logs.glob("*.monitor.csv"):
        old.unlink()
    venv = make_vec_env(TEEnv, n_envs=tcfg["n_envs"], seed=seed, env_kwargs=env_kwargs, monitor_dir=str(logs))

    algo = spec["algo"]
    model = ALGOS[algo]("MlpPolicy", venv, seed=seed, device="cpu", verbose=0, **HYPERPARAMS[algo])
    t0 = time.time()
    model.learn(total_timesteps=timesteps or tcfg["timesteps"])
    path = model_path(topology, name)
    model.save(path)
    meta = {"algo": algo, "env_kwargs": env_kwargs, "train_seconds": round(time.time() - t0, 1),
            "timesteps": timesteps or tcfg["timesteps"]}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


class DRLTEApp:
    """TE app backed by a trained policy: observes the network, outputs link weights,
    and routes each OD pair on its weighted shortest path."""

    def __init__(self, topology: str, name: str):
        path = model_path(topology, name)
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run: uv run sdnte train --topo {topology} --agent {name}")
        meta = json.loads(path.with_suffix(".json").read_text())
        self.name = name
        self.model = ALGOS[meta["algo"]].load(path, device="cpu")
        self.weight_range = tuple(meta["env_kwargs"]["weight_range"])

    def decide(self, state: NetworkState) -> Routing:
        obs = build_observation(state.topology, state.traffic_matrix, state.link_util)
        action, _ = self.model.predict(obs, deterministic=True)
        weights = action_to_weights(np.asarray(action), self.weight_range)
        return shortest_path_routing(state.topology, weights)
