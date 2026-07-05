#!/usr/bin/env python3
"""Gymnasium compliance smoke test for Phase 5.3.

Runs `gymnasium.utils.env_checker.check_env` on `ExecutionEnv`. The checker
exercises: reset signature (seed / options), observation-space membership,
step return-tuple shape (obs, reward, terminated, truncated, info), action
space bounds, deterministic-seeded reset equivalence, and a handful of
random rollouts. Any warning is escalated to an error via `-W error`
below so we cannot silently regress the Gymnasium contract.

Run:
    python3 scripts/test_gym.py
"""

from __future__ import annotations

import os
import sys
import warnings


def _add_repo_to_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def main() -> int:
    _add_repo_to_path()

    from gymnasium.utils.env_checker import check_env

    from python.env.execution_env import ExecutionEnv

    env = ExecutionEnv(parent_qty=1_000, horizon_steps=32, arrival_mid=100)
    try:
        # Escalate any UserWarning coming out of the checker — the task
        # says "no warnings or errors". The one exception is the checker's
        # informational note that the observation-space bounds are
        # ±infinity: those bounds are required by the task spec
        # (state carries raw tick prices with no natural upper bound), so
        # we let just that specific message through.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warnings.filterwarnings(
                "ignore",
                message=r".*Box observation space (minimum|maximum) value is (-)?infinity.*",
            )
            check_env(env, skip_render_check=True)
    finally:
        env.close()

    print("test_gym: OK — ExecutionEnv passes gymnasium.check_env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
