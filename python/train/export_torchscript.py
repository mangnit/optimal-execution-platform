"""Export the trained SAC actor to TorchScript for the C++ engine.

The C++ execution engine loads the deterministic policy via LibTorch, so
we only need the actor network — not the critic, replay buffer, or
optimizer state that live inside the SB3 `.zip`. We wrap the SB3 actor
in a thin `nn.Module` that always requests the deterministic action, then
trace it with a dummy observation matching `ExecutionEnv`'s 8-dim state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

from stable_baselines3 import SAC


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))


_OBS_DIM = 8  # ExecutionEnv observation_space shape.


class _DeterministicActor(nn.Module):
    """Freeze the SAC actor to its mean action so trace sees a pure tensor op."""

    def __init__(self, actor: nn.Module) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, deterministic=True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export SAC actor to TorchScript.")
    p.add_argument(
        "--model-path",
        type=Path,
        default=_REPO_ROOT / "models" / "sac_oep_baseline.zip",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=_REPO_ROOT / "models" / "sac_policy.pt",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.model_path.exists():
        print(f"[export] model not found: {args.model_path}", file=sys.stderr)
        return 1

    # SB3 needs a device; CPU keeps the exported graph portable.
    model = SAC.load(str(args.model_path), device="cpu")
    actor = model.policy.actor.eval()

    wrapped = _DeterministicActor(actor).eval()
    dummy_obs = torch.zeros(1, _OBS_DIM, dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, dummy_obs)

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.out_path))

    if not args.out_path.exists():
        print(f"[export] save failed: {args.out_path}", file=sys.stderr)
        return 1

    size_kb = args.out_path.stat().st_size / 1024.0
    print(f"[export] wrote {args.out_path} ({size_kb:.1f} KiB)")

    # Sanity check: reload and run the traced graph to catch a broken export
    # before the C++ side ever sees the file.
    reloaded = torch.jit.load(str(args.out_path))
    with torch.no_grad():
        out = reloaded(dummy_obs)
    print(f"[export] reload OK — output shape {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
