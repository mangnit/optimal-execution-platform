#!/usr/bin/env python3
"""Smoke test for the Phase 5.2 pybind11 bridge.

Imports the `oep_env` module built by CMake, spins up a SimEnv (which
brings up the same OrderBook + SpscRing + TelemetryPublisher + EventRelay
+ InMemoryKdbLogger stack `sim_runner` uses), and drives `step()` in a
loop with a couple of trivial policies:

  * TWAP-like: constant size fraction 1/horizon, passive (aggression 0).
  * All-in aggressive: full remaining, cross the spread every step.

The goal is to confirm the module imports, the C++ engine advances, no
SPSC drops are observed, and the reward stays finite. It is not an RL
training loop — the SB3 Gymnasium wrapper lands in a later step.

Run:
    python3 scripts/test_bridge.py
"""

from __future__ import annotations

import os
import sys

import numpy as np


def _import_oep_env():
    # Prefer an explicit OEP_ENV_DIR env var so an out-of-tree build can be
    # pointed at. Otherwise search the default CMake build directory.
    candidates = []
    env_dir = os.environ.get("OEP_ENV_DIR")
    if env_dir:
        candidates.append(env_dir)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.join(repo_root, "build"))
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
    try:
        import oep_env  # noqa: E402
    except ImportError as e:  # noqa: F841
        print(
            "test_bridge: could not import oep_env. Build it first:\n"
            "    cmake -S . -B build -G Ninja -DOEP_BUILD_PYBIND=ON \\\n"
            "        -DPython_EXECUTABLE=$(which python3)\n"
            "    cmake --build build --target oep_env\n"
            f"(searched: {candidates})",
            file=sys.stderr,
        )
        raise
    return oep_env


def _roll_episode(env, policy_name: str, action_fn):
    obs = env.reset(seed=0xDEADBEEF)
    assert obs.dtype == np.float32
    print(f"[{policy_name}] initial state: {obs}")
    total_reward = 0.0
    steps = 0
    while not env.done():
        action = action_fn(obs, env)
        r = env.step(action)
        obs = env.state()
        total_reward += r
        steps += 1
        assert np.isfinite(r), f"reward not finite: {r}"
    print(
        f"[{policy_name}] steps={steps} "
        f"total_reward={total_reward:.4f} bps "
        f"remaining={env.remaining()} "
        f"qty_filled={env.qty_filled()} "
        f"avg_fill_price={env.avg_fill_price():.4f} "
        f"processed={env.processed()} "
        f"dropped={env.dropped_messages()}"
    )
    return total_reward, steps


def main() -> int:
    oep_env = _import_oep_env()

    parent_qty = 1_000
    horizon = 32
    arrival_mid = 100

    env = oep_env.SimEnv(
        parent_qty=parent_qty,
        horizon_steps=horizon,
        seed=0xC0FFEEBABE,
        arrival_mid=arrival_mid,
    )
    print(
        f"SimEnv ready: state_dim={oep_env.SimEnv.state_dim} "
        f"action_dim={oep_env.SimEnv.action_dim}"
    )

    def twap_passive(_obs, _env):
        # Constant slice, post at the far side of the touch.
        return np.array([1.0 / horizon, 0.0], dtype=np.float32)

    def all_in_aggressive(_obs, env):
        # Sweep everything we have every step (aggression >= 0.5 crosses).
        return np.array([1.0, 1.0], dtype=np.float32)

    r_twap, s_twap = _roll_episode(env, "twap_passive", twap_passive)
    r_agg, s_agg = _roll_episode(env, "all_in_aggressive", all_in_aggressive)

    # Basic invariants a bridge run should always satisfy. Liquidity in the
    # synthetic exogenous flow is deliberately thin, so we do NOT assert that
    # any particular policy closes the parent — that is the RL agent's job in
    # a later phase. What we DO assert here is bridge integrity.
    assert env.dropped_messages() == 0, (
        f"SPSC ring dropped {env.dropped_messages()} messages — the ring is "
        "either too small or the relay is stalling."
    )
    assert 1 <= s_twap <= horizon and 1 <= s_agg <= horizon, (
        "episode length must stay bounded by the horizon"
    )
    assert np.isfinite(r_twap) and np.isfinite(r_agg)

    print("test_bridge: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
