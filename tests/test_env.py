import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from sdnte.env import TEEnv, action_to_weights


@pytest.mark.parametrize("topo", ["toy6", "abilene"])
def test_env_passes_gymnasium_checker(topo):
    check_env(TEEnv(topo), skip_render_check=True)


def test_equal_weights_match_ospf_hop():
    # All weights equal == hop-count routing, so relative MLU gain is zero.
    env = TEEnv("abilene", load_range=(0.5, 0.5), failure_prob=0.0)
    env.reset(seed=0)
    for _ in range(5):
        _, reward, _, _, info = env.step(np.zeros(env.action_space.shape))
        assert info["mlu"] == pytest.approx(info["ospf_mlu"])
        assert reward == pytest.approx(0.0)


def test_episode_length_and_failure_visible_in_observation():
    env = TEEnv("toy6", episode_steps=10, failure_prob=1.0)
    obs, _ = env.reset(seed=3)
    L = env.topo.num_links
    assert obs[L : 2 * L].min() == 1.0  # all links up at the start
    seen_failure, steps = False, 0
    terminated = False
    while not terminated:
        obs, _, terminated, _, _ = env.step(env.action_space.sample())
        steps += 1
        if not terminated and obs[L : 2 * L].min() == 0.0:
            seen_failure = True
    assert steps == 10 and seen_failure


def test_action_to_weights_range():
    w = action_to_weights(np.array([-1.0, 0.0, 1.0, 5.0]), (1.0, 20.0))
    np.testing.assert_allclose(w, [1.0, 10.5, 20.0, 20.0])
