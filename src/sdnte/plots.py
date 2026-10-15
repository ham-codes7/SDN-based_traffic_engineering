"""Report figures from results/*.csv into figures/ (owner P3, week 4).

Planned figures (PNG + PDF, consistent method colours across all figures):
1. MLU vs load per method (line, mean +- 95% CI over seeds), one panel per topology
2. Packet loss and average delay vs load
3. MLU by traffic pattern at load 0.9 (grouped bars)
4. Link-failure timeline: MLU per step around the failure
5. Generalization: trained on gravity, tested on hotspot/bursty
6. PPO vs A2C training reward curves
7. Gap to LP optimum and decision time per method (table)
"""
