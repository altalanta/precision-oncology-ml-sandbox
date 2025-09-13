from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AssayEnv:
    """Cost-aware bandit-like environment.

    - K assays with per-assay cost and stochastic information gain ~ N(theta_k, 1)
    - Episode ends when budget is exhausted; reward is gain - cost
    """

    K: int
    budget: int
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.costs = self.rng.integers(1, 4, size=self.K)
        self.theta = self.rng.normal(0.5, 0.3, size=self.K)
        self.reset()

    def reset(self) -> np.ndarray:
        self.remaining = self.budget
        self.total_reward = 0.0
        # Observation: remaining budget and normalized costs
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.concatenate([[self.remaining], self.costs / self.costs.max()]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < self.K
        cost = int(self.costs[action])
        if self.remaining - cost < 0:
            # invalid action: no budget; end episode with penalty
            return self._obs(), -1.0, True, {"invalid": True}
        self.remaining -= cost
        gain = float(self.rng.normal(self.theta[action], 1.0))
        reward = gain - cost * 0.1
        self.total_reward += reward
        done = self.remaining <= 0
        return self._obs(), reward, done, {"gain": gain, "cost": cost}

