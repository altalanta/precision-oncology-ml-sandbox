import numpy as np

from pon_ml.rl.env import AssayEnv


def test_env_step_and_reset():
    env = AssayEnv(K=3, budget=5, seed=1)
    obs = env.reset()
    assert obs.shape[0] == 1 + env.K
    obs2, reward, done, info = env.step(0)
    assert not np.isnan(reward)
    assert obs2.shape == obs.shape

