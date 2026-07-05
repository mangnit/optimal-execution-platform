"""SAC training loop for the optimal-execution agent (docs/architecture.md §5.4).

The C++ engine step releases the GIL (docs/architecture.md P5+), so `SubprocVecEnv`
spins up N independent SimEnv pipelines that make real parallel progress.
This script is the smoke-test entry point: it wires SB3 SAC to
`ExecutionEnv`, trains for a small number of steps, and drops both a
Tensorboard trace and a `.zip` model artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from env.execution_env import ExecutionEnv


_DEFAULT_TRAIN_SEED = 0xC0FFEEBABE
_DEFAULT_EVAL_SEED = 0xDEADBEEF


def _make_env(rank: int, seed: int):
    """Return a thunk that Gymnasium/SB3 will call inside the subprocess.

    Each rank gets a distinct seed so SubprocVecEnv workers explore
    decorrelated trajectories rather than N copies of the same episode.
    """
    def _init() -> Monitor:
        env = ExecutionEnv(seed=seed + rank)
        return Monitor(env)

    return _init


def _build_train_env(n_envs: int, seed: int) -> SubprocVecEnv:
    # SubprocVecEnv is the whole point of P5.2 GIL release — use it even
    # for smoke tests so a regression in the bridge shows up here.
    return SubprocVecEnv([_make_env(i, seed) for i in range(n_envs)])


def _build_eval_env(seed: int) -> DummyVecEnv:
    # Deterministic single-process env keeps eval numbers reproducible.
    return DummyVecEnv([_make_env(0, seed)])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC smoke-test trainer for OEP.")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--total-timesteps", type=int, default=10_000)
    p.add_argument("--eval-freq", type=int, default=2_000)
    p.add_argument("--n-eval-episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=_DEFAULT_TRAIN_SEED)
    p.add_argument("--eval-seed", type=int, default=_DEFAULT_EVAL_SEED)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=_REPO_ROOT / "logs",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=_REPO_ROOT / "models" / "sac_oep_baseline.zip",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_model_dir = args.log_dir / "eval_best"
    best_model_dir.mkdir(parents=True, exist_ok=True)

    train_env = _build_train_env(args.n_envs, args.seed)
    eval_env = _build_eval_env(args.eval_seed)

    try:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            seed=args.seed & 0x7FFFFFFF,
            tensorboard_log=str(args.log_dir),
            # Small buffer / batch keeps the smoke test light; production
            # runs override these via a config sweep, not by editing here.
            buffer_size=50_000,
            batch_size=256,
            learning_starts=1_000,
            learning_rate=1e-4,
            ent_coef="auto",
        )

        # SB3 divides eval_freq by n_envs internally for SubprocVecEnv.
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(best_model_dir),
            log_path=str(args.log_dir / "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=eval_callback,
            tb_log_name="sac_oep",
            progress_bar=False,
        )

        model.save(str(args.model_path))
        print(f"[train_sac] saved model to {args.model_path}")
        print(f"[train_sac] tensorboard logs at {args.log_dir}")
    finally:
        train_env.close()
        eval_env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
