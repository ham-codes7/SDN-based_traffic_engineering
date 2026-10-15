import networkx as nx
import numpy as np
import pytest

from sdnte.controller import SDNController
from sdnte.experiments import run_episode
from sdnte.network import evaluate, link_loads
from sdnte.routing import make_te_app
from sdnte.routing.baselines import OSPFHop, shortest_path_routing
from sdnte.topology import Topology
from sdnte.traffic import PATTERNS, generate_sequence

EXPECTED = {"toy6": (6, 8), "abilene": (12, 15), "nsfnet": (14, 21)}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_topologies_load(name):
    topo = Topology.load(name)
    nodes, undirected = EXPECTED[name]
    assert topo.num_nodes == nodes
    assert topo.num_links == 2 * undirected
    assert nx.is_strongly_connected(topo.graph())


@pytest.mark.parametrize("pattern", PATTERNS)
def test_traffic_shape_and_scaling(pattern):
    topo = Topology.load("abilene")
    seq = generate_sequence(topo, pattern, steps=10, load=0.7, seed=1)
    assert seq.shape == (10, 12, 12)
    assert np.all(seq >= 0)
    assert np.all(seq[:, np.arange(12), np.arange(12)] == 0)
    # Load scaling: the mean TM under hop-count routing puts the busiest link at 0.7.
    routing = shortest_path_routing(topo)
    mlu = (link_loads(topo, routing, seq.mean(axis=0)) / topo.capacities).max()
    assert mlu == pytest.approx(0.7)


def test_traffic_is_reproducible():
    topo = Topology.load("toy6")
    a = generate_sequence(topo, "bursty", 5, 0.5, seed=3)
    b = generate_sequence(topo, "bursty", 5, 0.5, seed=3)
    np.testing.assert_array_equal(a, b)


def test_ospf_hop_paths_are_min_hop():
    topo = Topology.load("nsfnet")
    routing = shortest_path_routing(topo)
    lengths = dict(nx.all_pairs_shortest_path_length(topo.graph()))
    assert len(routing) == len(topo.od_pairs)
    for (s, d), [(path, frac)] in routing.items():
        assert frac == 1.0 and path[0] == s and path[-1] == d
        assert len(path) - 1 == lengths[s][d]


def test_link_load_conservation():
    # Total link load = sum over pairs of demand x path length.
    topo = Topology.load("abilene")
    tm = generate_sequence(topo, "gravity", 1, 0.5, seed=0)[0]
    routing = shortest_path_routing(topo)
    expected = sum(tm[s, d] * (len(p) - 1) for (s, d), [(p, _)] in routing.items())
    assert link_loads(topo, routing, tm).sum() == pytest.approx(expected)


def test_no_loss_below_capacity_and_loss_above():
    topo = Topology.load("toy6")
    routing = shortest_path_routing(topo)
    tm = generate_sequence(topo, "uniform", 1, 1.0, seed=0)[0]
    low = evaluate(topo, routing, tm * 0.5 / evaluate(topo, routing, tm).mlu)
    assert low.mlu == pytest.approx(0.5)
    assert low.loss_pct == pytest.approx(0.0)
    high = evaluate(topo, routing, tm * 2.0 / evaluate(topo, routing, tm).mlu)
    assert high.loss_pct > 0
    assert high.avg_delay_ms > low.avg_delay_ms


def test_controller_installs_rules_and_handles_failure():
    topo = Topology.load("toy6")
    tm_seq = generate_sequence(topo, "gravity", 6, 0.6, seed=0)
    ctrl = SDNController(topo, OSPFHop())
    metrics, decision_ms = ctrl.step(tm_seq[0])
    assert ctrl.num_rules > 0 and decision_ms >= 0
    np.testing.assert_allclose(ctrl.link_util, metrics.link_util)
    # A->D chord fails: OSPF reroutes over the ring, nothing is sent on the dead link.
    ctrl.fail_link(0, 3)
    metrics, _ = ctrl.step(tm_seq[1])
    assert metrics.link_util[topo.link_index[(0, 3)]] == 0
    assert metrics.loss_pct == pytest.approx(0.0)


def test_run_episode_ospf_hop_end_to_end():
    topo = Topology.load("toy6")
    df = run_episode(topo, make_te_app("ospf_hop"), generate_sequence(topo, "gravity", 8, 0.6, seed=0), fail_at=4, fail_link=(1, 4))
    assert len(df) == 8
    assert set(df["method"]) == {"ospf_hop"}
    assert df["mlu"].between(0, 2).all()
