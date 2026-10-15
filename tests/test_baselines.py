import networkx as nx
import numpy as np
import pytest

from sdnte.controller import NetworkState
from sdnte.network import evaluate
from sdnte.routing import BASELINES, make_te_app
from sdnte.topology import Topology
from sdnte.traffic import generate_sequence


def _state(topo, tm):
    return NetworkState(topo, tm, np.zeros(topo.num_links), 0)


@pytest.mark.parametrize("method", sorted(BASELINES))
@pytest.mark.parametrize("topo_name", ["toy6", "abilene"])
def test_routing_is_valid(method, topo_name):
    topo = Topology.load(topo_name)
    tm = generate_sequence(topo, "hotspot", 1, 0.9, seed=2)[0]
    routing = make_te_app(method).decide(_state(topo, tm))
    assert set(routing) == set(topo.od_pairs)
    for (s, d), splits in routing.items():
        assert sum(f for _, f in splits) == pytest.approx(1.0)
        for path, frac in splits:
            assert path[0] == s and path[-1] == d and frac > 0
            topo.path_links(path)  # raises KeyError if a hop is not a link


def test_ecmp_splits_over_all_min_hop_paths():
    topo = Topology.load("toy6")
    routing = make_te_app("ecmp").decide(_state(topo, np.ones((6, 6))))
    # A(0) -> C(2): A-B-C, A-D-C are both 2 hops.
    paths = [p for p, _ in routing[(0, 2)]]
    assert sorted(paths) == sorted(nx.all_shortest_paths(topo.graph(), 0, 2))
    assert all(f == pytest.approx(1 / len(paths)) for _, f in routing[(0, 2)])


@pytest.mark.parametrize("topo_name", ["toy6", "abilene", "nsfnet"])
@pytest.mark.parametrize("pattern", ["gravity", "hotspot"])
def test_lp_is_a_lower_bound_on_mlu(topo_name, pattern):
    topo = Topology.load(topo_name)
    for tm in generate_sequence(topo, pattern, 3, 0.9, seed=5):
        mlus = {m: evaluate(topo, make_te_app(m).decide(_state(topo, tm)), tm).mlu for m in BASELINES}
        for m, mlu in mlus.items():
            assert mlus["lp_optimal"] <= mlu + 1e-6, (m, mlus)


def test_adaptive_methods_beat_ospf_hop_under_hotspot_load():
    topo = Topology.load("abilene")
    tm_seq = generate_sequence(topo, "hotspot", 5, 0.9, seed=0)
    mean = {
        m: np.mean([evaluate(topo, make_te_app(m).decide(_state(topo, tm)), tm).mlu for tm in tm_seq])
        for m in ("ospf_hop", "greedy", "lp_optimal")
    }
    assert mean["greedy"] < mean["ospf_hop"]
    assert mean["lp_optimal"] < mean["greedy"]


def test_baselines_route_around_failed_link():
    topo = Topology.load("abilene")
    topo.fail_link(topo.node_names.index("CHINng"), topo.node_names.index("NYCMng"))
    tm = generate_sequence(topo, "gravity", 1, 0.6, seed=0)[0]
    for m in BASELINES:
        metrics = evaluate(topo, make_te_app(m).decide(_state(topo, tm)), tm)
        assert metrics.link_util[~topo.up].sum() == 0
