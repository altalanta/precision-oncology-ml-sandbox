from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from pon_ml.rl.env import AssayEnv
from pon_ml.seeds import set_seeds
from pon_ml.tracking import init_mlflow, log_metrics, mlflow


def q_learning(env: AssayEnv, episodes: int, epsilon: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    Q = np.zeros(env.K)
    returns = []
    for ep in range(episodes):
        env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                a = int(rng.integers(0, env.K))
            else:
                a = int(np.argmax(Q))
            _, r, done, _ = env.step(a)
            Q[a] = 0.9 * Q[a] + 0.1 * r
        returns.append(env.total_reward)
    return {"avg_return": float(np.mean(returns)), "Q": Q.tolist()}


@hydra.main(config_path="../../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # pragma: no cover
    set_seeds(cfg.seed)
    init_mlflow(cfg.mlflow.uri, cfg.mlflow.experiment)
    env = AssayEnv(K=cfg.rl.assays, budget=cfg.rl.budget, seed=cfg.seed)
    with mlflow.start_run(run_name="rl_q_learning"):
        metrics = q_learning(env, episodes=cfg.rl.episodes, epsilon=cfg.rl.epsilon, seed=cfg.seed)
        log_metrics(metrics)


if __name__ == "__main__":
    main()

