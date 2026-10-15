# Report outline (IEEE conference style)

1. **Abstract**: problem, approach (DRL link-weight TE in SDN), key result (MLU / loss improvement vs OSPF and ECMP, gap to LP optimum).
2. **Introduction**: the TE problem, limits of static OSPF, why SDN's centralized view and why DRL; contributions list.
3. **Related Work**: group the 10 assigned papers:
   - Segment-routing TE: [1] recommendation-based acceleration, [2] adaptive DRL SR-TE.
   - Supervised and representation learning for routing: [3] PROM, [4] contrastive learning for hybrid SDN, [5] MTEAL multi-QoS SD-WAN.
   - DRL routing in SDN: [6] DRL + betweenness centrality, [7] knowledge transfer RL, [8] bandwidth-demand DRL routing, [9] LCA congestion avoidance, [10] predictive routing for AAV networks.
4. **System Architecture**: controller (flow tables, port stats, northbound TE API), flow-level data-plane model, TE apps.
5. **DRL Formulation**: MDP (observation, action = link weights, reward = -MLU - alpha * loss), PPO, A2C.
6. **Experimental Setup**: topologies, traffic patterns, load levels, failure and generalization scenarios, baselines, metrics, hyperparameters, seeds.
7. **Results**: MLU / loss / delay vs load; by traffic pattern; failure recovery; generalization; PPO vs A2C; gap to LP optimum; decision time.
8. **Discussion & Limitations**: flow-level simulation vs Mininet / packet level; per-topology policies; synthetic traffic.
9. **Conclusion & Future Work**: GNN-based policies for unseen topologies, Mininet + Ryu deployment, real traffic traces.
