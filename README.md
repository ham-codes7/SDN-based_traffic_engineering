# SDN-Based Intelligent Traffic Engineering Using Deep Reinforcement Learning

CN coursework project. A simulated SDN controller hosts a traffic-engineering (TE) app. A DRL agent (PPO) learns link weights that adapt routing to the current traffic. It is compared against OSPF, ECMP, a greedy heuristic, an LP-optimal bound and A2C under varying load, traffic patterns, link failures and topologies.

## Setup (macOS, Apple Silicon or Intel)

Everything runs natively on macOS. You don't need Mininet, Docker or a VM.

```bash
brew install uv          # Python package/env manager
uv sync                  # creates .venv with Python 3.11 and installs all dependencies
```

## Usage

```bash
uv run sdnte topologies                          # list topologies
uv run sdnte simulate --topo toy6                # OSPF-hop on toy6, gravity traffic, load 0.6
uv run sdnte simulate --topo abilene --method ospf_invcap --pattern hotspot --load 0.9
uv run pytest                                    # run the tests
uv run streamlit run dashboard/app.py            # dashboard (placeholder for now)
```

`--load` is relative to hop-count OSPF: at `1.0`, the time-averaged traffic exactly saturates the busiest link under OSPF-hop routing. Each `simulate` run saves per-step metrics to `results/`.

## Project layout

| Path | What it is | Status |
|---|---|---|
| `data/topologies/` | toy6, Abilene, NSFNet as JSON | done |
| `src/sdnte/topology.py` | Topology loading, link indexing, link failures | done |
| `src/sdnte/traffic.py` | Traffic matrix generators (uniform, gravity, hotspot, bursty) + load scaling | done |
| `src/sdnte/network.py` | Flow-level data plane: link loads, MLU, loss, delay, throughput | done |
| `src/sdnte/controller.py` | SDN controller: flow tables, port stats, TE app interface | done |
| `src/sdnte/routing/baselines.py` | OSPF-hop and OSPF-invcap done; ECMP and greedy TODO | partial |
| `src/sdnte/routing/optimal.py` | LP-optimal min-MLU bound | TODO (P3) |
| `src/sdnte/env.py`, `agents.py` | Gymnasium env, PPO/A2C training, DRL TE app | TODO (P2) |
| `src/sdnte/experiments.py` | `run_episode` done; full grid TODO | partial |
| `src/sdnte/plots.py` | Report figures | TODO (P3) |
| `dashboard/` | Streamlit dashboard | TODO (P4) |
| `configs/default.yaml` | Experiment grid and training settings | done |

## How a control interval works

1. The traffic generator produces this interval's traffic matrix.
2. The controller passes the TE app a `NetworkState`: the topology, the measured demand, and last interval's link utilization from port stats.
3. The TE app returns a routing: paths per source–destination pair, with split fractions.
4. The controller installs flow rules on each switch.
5. The data-plane model carries the traffic and reports metrics.

To add a TE method, write a class with a `name` and a `decide(state) -> Routing` method, then register it in `src/sdnte/routing/__init__.py`.
