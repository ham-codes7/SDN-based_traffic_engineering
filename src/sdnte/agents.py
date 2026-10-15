"""DRL agents: training with Stable-Baselines3 and the DRL TE app (owner P2, week 3).

Planned API:
- train(topo_name, algo="ppo" | "a2c", timesteps, n_envs, seed) -> saves
  models/{topo}_{algo}.zip and training logs (SB3 Monitor CSV) for reward curves
- DRLTEApp(model_path): TE app whose decide(state) builds the TEEnv observation from
  the NetworkState, calls model.predict(obs, deterministic=True), maps the action to
  link weights and returns shortest_path_routing(state.topology, weights)
- register "ppo" and "a2c" so routing.make_te_app can build them for experiments
"""
