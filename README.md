# SDN-Based Intelligent Traffic Engineering Using Deep Reinforcement Learning

CN coursework project. A simulated SDN controller hosts a traffic-engineering (TE) app. A DRL agent (PPO) learns link weights that adapt routing to the current traffic. It is compared against OSPF, ECMP, a greedy heuristic, an LP-optimal bound and A2C under varying load, traffic patterns, link failures and topologies.

## Setup (macOS, Apple Silicon or Intel)

Everything runs natively on macOS. You don't need Mininet, Docker or a VM.

```bash
brew install uv          # Python package/env manager
uv sync                  # creates .venv with Python 3.11 and installs all dependencies
```

## Reproduce everything

```bash
uv run sdnte train       # trains ppo, a2c, ppo_gravity on Abilene and NSFNet (~4-7 min each)
uv run sdnte run-all     # full experiment grid -> results/ (add --quick for a ~2 min check)
uv run sdnte plots       # report figures -> figures/, summary table -> results/summary_load0.9.md
uv run streamlit run dashboard/Home.py
```

Trained models, results and figures are generated locally and not committed.

## Other commands

```bash
uv run sdnte topologies                                    # list topologies
uv run sdnte simulate --topo abilene --method ppo --pattern hotspot --load 0.9
uv run sdnte train --topo abilene --agent ppo --timesteps 50000
uv run pytest                                              # tests
```

`--load` is relative to hop-count OSPF: at `1.0`, the time-averaged traffic exactly saturates the busiest link under OSPF-hop routing. Methods: `ospf_hop`, `ospf_invcap`, `ecmp`, `greedy`, `lp_optimal`, plus trained agents `ppo`, `a2c`, `ppo_gravity`.

## Dashboard

| Page | What it shows |
|---|---|
| Home | Project overview, methods, trained agents |
| Simulate | Two methods side by side on identical traffic: network map coloured by link utilization, step slider, optional link failure, metrics over time |
| Results | MLU / loss / delay vs load and by traffic pattern, summary table (reduction vs OSPF, gap to LP optimum) |
| Failure | Extra utilization caused by a link failure, compared against the same traffic without it |
| Training | Reward curves and training settings of each agent |

## How it works

| Path | What it is |
|---|---|
| `data/topologies/` | toy6, Abilene, NSFNet as JSON |
| `src/sdnte/topology.py` | Topology loading, link indexing, link failures |
| `src/sdnte/traffic.py` | Traffic matrix generators (uniform, gravity, hotspot, bursty) + load scaling |
| `src/sdnte/network.py` | Flow-level data plane: link loads, MLU, loss, M/M/1 delay, throughput |
| `src/sdnte/controller.py` | SDN controller: flow tables, port stats, TE app interface |
| `src/sdnte/routing/` | Baselines: OSPF (hop, inverse capacity), ECMP, greedy least-loaded, LP-optimal |
| `src/sdnte/env.py` | Gymnasium environment for training |
| `src/sdnte/agents.py` | PPO/A2C training and the deployed DRL TE app |
| `src/sdnte/experiments.py` | Experiment grid runner |
| `src/sdnte/plots.py` | Report figures and summary table |
| `dashboard/` | Streamlit dashboard |
| `configs/default.yaml` | Experiment grid and training settings |

Each control interval:
1. The traffic generator produces this interval's traffic matrix.
2. The controller passes the TE app a `NetworkState`: the topology (including failed links), the measured demand, and last interval's link utilization from port stats.
3. The TE app returns a routing: paths per source–destination pair, with split fractions.
4. The controller installs flow rules on each switch.
5. The data-plane model carries the traffic and reports metrics.

**DRL formulation.**
- **Observation:** previous link utilization, link up/down mask and the normalized traffic matrix.
- **Action:** a weight for every directed link. Each pair is routed on its weighted shortest path.
- **Reward:** relative MLU reduction versus hop-count OSPF on the same traffic, minus a packet-loss penalty.
- **Training:** episodes sample a traffic pattern, a load and an optional link failure, so one policy covers all test conditions.

To add a TE method, write a class with a `name` and a `decide(state) -> Routing` method, then register it in `src/sdnte/routing/__init__.py`.
