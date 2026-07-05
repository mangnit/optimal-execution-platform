"""Gymnasium wrapper around the C++ SimEnv (docs/architecture.md §5.4).

The C++ bridge (`oep_env.SimEnv`) already owns the OrderBook →
TelemetryPublisher → SpscRing → EventRelay pipeline and exposes a flat
float32 state + reward interface with the GIL released across the engine
step (docs/architecture.md P5+ rule). This wrapper is a thin Gymnasium adapter so
Stable-Baselines3 can drive it.

Contract (kept minimal to avoid per-step Python-side allocations):
  * `reset` and `step` return the zero-copy numpy view exposed by the C++
    side. The state buffer is `alignas(64) float[8]` on the C++ side; the
    view lives as long as the wrapper (which owns the SimEnv). Callers
    that need to hold state across steps should copy.
  * `action` is a length-2 float32 numpy array. Gymnasium samples from
    the Box action space directly, so no allocation happens on the RL
    side either.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

import gymnasium as gym
from gymnasium import spaces


def _ensure_oep_env_on_path() -> None:
    env_dir = os.environ.get("OEP_ENV_DIR")
    candidates: list[str] = []
    if env_dir:
        candidates.append(env_dir)
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    candidates.append(os.path.join(repo_root, "build"))
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


_ensure_oep_env_on_path()
import oep_env  # noqa: E402


_DEFAULT_SEED = 0xC0FFEEBABE


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        parent_qty: int = 1000,
        horizon_steps: int = 32,
        arrival_mid: int = 100,
        seed: int = _DEFAULT_SEED,
    ) -> None:
        super().__init__()
        self._parent_qty = int(parent_qty)
        self._horizon_steps = int(horizon_steps)
        self._arrival_mid = int(arrival_mid)
        self._default_seed = int(seed) & 0xFFFFFFFFFFFFFFFF
        self._sim = oep_env.SimEnv(
            parent_qty=self._parent_qty,
            horizon_steps=self._horizon_steps,
            seed=self._default_seed,
            arrival_mid=self._arrival_mid,
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        seed_val = (
            self._default_seed if seed is None else int(seed) & 0xFFFFFFFFFFFFFFFF
        )
        obs = self._sim.reset(seed=seed_val)
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # The C++ side declares the action buffer with `py::array::forcecast`,
        # so a non-float32 array is coerced in-flight without allocating here
        # when Gymnasium already hands us a float32 sample.
        reward = self._sim.step(action)
        obs = self._sim.state()
        terminated = self._sim.done()
        truncated = False
        return obs, float(reward), terminated, truncated, {}

    def close(self) -> None:
        # Drop the SimEnv so its dtor stops the relay thread. Guard against
        # a second close() coming from Gymnasium's finalizer.
        self._sim = None
