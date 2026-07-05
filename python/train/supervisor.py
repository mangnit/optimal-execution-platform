"""Training-lifecycle supervisor around `train_sac.py`.

Launches the SB3 SAC trainer as a subprocess and watches two failure
modes that a bare `python train_sac.py` invocation would silently
tolerate:

  1. **RAM growth** — psutil samples RSS of the trainer + its children
     every few seconds. If RSS grows by more than 500 MB across a
     rolling 10-minute window we treat that as a leak (SubprocVecEnv +
     LibTorch buffers make slow leaks plausible), log CRITICAL, dump a
     memory snapshot to `logs/supervisor/`, and kill the process tree.

  2. **Reward plateau** — we tail the Tensorboard event files under
     `logs/` with `event_accumulator` and read `rollout/ep_rew_mean`.
     If it fails to strictly improve over its previous best for 10
     consecutive eval intervals, we early-stop the trainer via a clean
     SIGTERM (SIGKILL as fallback).

A `logs/supervisor/status.txt` heartbeat is rewritten each poll with
the latest step count and RSS so an operator can `watch cat` it.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_sac.py"

# Memory-leak detector.
_MEM_LEAK_THRESHOLD_MB = 500.0
_MEM_WINDOW_SECONDS = 10 * 60

# Reward-plateau detector.
_PLATEAU_PATIENCE = 50
_REWARD_TAG = "rollout/ep_rew_mean"
_STEP_TAG_CANDIDATES = ("time/total_timesteps", "rollout/ep_rew_mean")

_POLL_INTERVAL_SECONDS = 5.0
_SIGTERM_GRACE_SECONDS = 30.0


@dataclass
class MemorySample:
    monotonic_ts: float
    rss_mb: float


def _configure_logging(supervisor_dir: Path) -> logging.Logger:
    supervisor_dir.mkdir(parents=True, exist_ok=True)
    log_path = supervisor_dir / "supervisor.log"
    logger = logging.getLogger("oep.supervisor")
    logger.setLevel(logging.INFO)
    # Idempotent: repeated runs in the same process (tests) mustn't stack handlers.
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_h = logging.FileHandler(log_path)
    file_h.setFormatter(fmt)
    stream_h = logging.StreamHandler(sys.stderr)
    stream_h.setFormatter(fmt)
    logger.addHandler(file_h)
    logger.addHandler(stream_h)
    return logger


def _tree_rss_mb(proc: psutil.Process) -> float:
    """Sum RSS of trainer + descendants.

    SubprocVecEnv forks N worker processes so the parent alone
    massively understates the trainer's real footprint.
    """
    total = 0
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except psutil.NoSuchProcess:
        return 0.0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total / (1024 * 1024)


def _dump_memory_snapshot(proc: psutil.Process, snapshot_path: Path) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# memory snapshot {datetime.utcnow().isoformat()}Z", ""]
    procs = [proc]
    try:
        procs.extend(proc.children(recursive=True))
    except psutil.NoSuchProcess:
        pass
    for p in procs:
        try:
            info = p.as_dict(attrs=["pid", "name", "cmdline", "memory_info", "status"])
            rss_mb = info["memory_info"].rss / (1024 * 1024) if info["memory_info"] else 0.0
            lines.append(
                f"pid={info['pid']} status={info['status']} rss_mb={rss_mb:.1f} "
                f"name={info['name']} cmd={' '.join(info['cmdline'] or [])}"
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    snapshot_path.write_text("\n".join(lines) + "\n")


def _latest_event_file(log_dir: Path) -> Optional[Path]:
    if not log_dir.exists():
        return None
    candidates = list(log_dir.rglob("events.out.tfevents.*"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_reward_series(event_path: Path) -> list[tuple[int, float]]:
    """Return `[(step, ep_rew_mean), ...]` from the newest event file.

    EventAccumulator is expensive to instantiate; we accept that cost
    every poll because event files are small during training and this
    keeps the code straightforward.
    """
    acc = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception:
        return []
    tags = acc.Tags().get("scalars", [])
    if _REWARD_TAG not in tags:
        return []
    return [(s.step, s.value) for s in acc.Scalars(_REWARD_TAG)]


def _write_status(
    status_path: Path,
    *,
    pid: Optional[int],
    step: Optional[int],
    rss_mb: float,
    best_reward: Optional[float],
    plateau_count: int,
    started_at: str,
) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    step_str = "?" if step is None else str(step)
    reward_str = "?" if best_reward is None else f"{best_reward:.4f}"
    body = (
        f"started_at   {started_at}\n"
        f"updated_at   {datetime.utcnow().isoformat()}Z\n"
        f"trainer_pid  {pid}\n"
        f"step         {step_str}\n"
        f"rss_mb       {rss_mb:.1f}\n"
        f"best_reward  {reward_str}\n"
        f"plateau/{_PLATEAU_PATIENCE}  {plateau_count}\n"
    )
    # Atomic swap so a concurrent `cat` never sees a half-written file.
    tmp = status_path.with_suffix(status_path.suffix + ".tmp")
    tmp.write_text(body)
    os.replace(tmp, status_path)


def _terminate(proc: subprocess.Popen, logger: logging.Logger, reason: str) -> None:
    if proc.poll() is not None:
        return
    logger.warning("terminating trainer (%s) — sending SIGTERM to pid=%d", reason, proc.pid)
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=_SIGTERM_GRACE_SECONDS)
        logger.info("trainer exited cleanly after SIGTERM")
        return
    except subprocess.TimeoutExpired:
        pass
    logger.error("trainer did not exit within %.0fs; sending SIGKILL", _SIGTERM_GRACE_SECONDS)
    try:
        proc.kill()
    except ProcessLookupError:
        return
    proc.wait()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training-lifecycle supervisor for SAC.")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=_REPO_ROOT / "logs",
        help="Tensorboard log directory (must match train_sac.py --log-dir).",
    )
    p.add_argument(
        "--supervisor-dir",
        type=Path,
        default=_REPO_ROOT / "logs" / "supervisor",
        help="Where status.txt, supervisor.log, and snapshots are written.",
    )
    # Everything after `--` gets forwarded to the trainer verbatim.
    p.add_argument("trainer_args", nargs=argparse.REMAINDER)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logger = _configure_logging(args.supervisor_dir)

    forwarded = args.trainer_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    cmd = [sys.executable, str(_TRAIN_SCRIPT), *forwarded]
    started_at = datetime.utcnow().isoformat() + "Z"
    logger.info("launching trainer: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT))
    ps_proc = psutil.Process(proc.pid)

    status_path = args.supervisor_dir / "status.txt"
    snapshot_path = (
        args.supervisor_dir
        / f"memory_snapshot_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.txt"
    )

    mem_window: deque[MemorySample] = deque()
    best_reward: Optional[float] = None
    last_reward_step: Optional[int] = None
    plateau_count = 0
    last_step: Optional[int] = None
    exit_code = 0

    try:
        while proc.poll() is None:
            now = time.monotonic()
            rss_mb = _tree_rss_mb(ps_proc)

            mem_window.append(MemorySample(now, rss_mb))
            while mem_window and now - mem_window[0].monotonic_ts > _MEM_WINDOW_SECONDS:
                mem_window.popleft()

            # Only judge growth once we actually have a full window — otherwise
            # a fast-ramping warmup would trip the detector.
            if (
                mem_window
                and now - mem_window[0].monotonic_ts >= _MEM_WINDOW_SECONDS
            ):
                window_min = min(s.rss_mb for s in mem_window)
                growth = rss_mb - window_min
                if growth > _MEM_LEAK_THRESHOLD_MB:
                    logger.critical(
                        "MEMORY LEAK DETECTED: RSS grew %.1f MB over %d-min window "
                        "(min=%.1f, now=%.1f MB) — killing trainer",
                        growth,
                        _MEM_WINDOW_SECONDS // 60,
                        window_min,
                        rss_mb,
                    )
                    _dump_memory_snapshot(ps_proc, snapshot_path)
                    logger.critical("wrote snapshot to %s", snapshot_path)
                    _terminate(proc, logger, "memory leak")
                    exit_code = 2
                    break

            event_file = _latest_event_file(args.log_dir)
            if event_file is not None:
                series = _read_reward_series(event_file)
                if series:
                    last_step, current_reward = series[-1]
                    if best_reward is None or current_reward > best_reward:
                        best_reward = current_reward
                        last_reward_step = last_step
                        plateau_count = 0
                    elif last_step != last_reward_step:
                        # Only count a new evaluation interval once — repeated
                        # polls between evals must not inflate `plateau_count`.
                        plateau_count += 1
                        last_reward_step = last_step
                        logger.info(
                            "no reward improvement at step %d (best=%.4f, now=%.4f) "
                            "[%d/%d]",
                            last_step,
                            best_reward,
                            current_reward,
                            plateau_count,
                            _PLATEAU_PATIENCE,
                        )
                        if plateau_count >= _PLATEAU_PATIENCE:
                            logger.warning(
                                "EARLY STOPPING: ep_rew_mean has not improved for "
                                "%d consecutive eval intervals (best=%.4f at step %d)",
                                _PLATEAU_PATIENCE,
                                best_reward,
                                last_step,
                            )
                            _terminate(proc, logger, "early stopping")
                            exit_code = 0
                            break

            _write_status(
                status_path,
                pid=proc.pid,
                step=last_step,
                rss_mb=rss_mb,
                best_reward=best_reward,
                plateau_count=plateau_count,
                started_at=started_at,
            )

            time.sleep(_POLL_INTERVAL_SECONDS)

        if proc.poll() is not None and exit_code == 0:
            exit_code = proc.returncode or 0
            logger.info("trainer exited on its own with code %d", exit_code)
    except KeyboardInterrupt:
        logger.warning("supervisor interrupted — forwarding SIGINT to trainer")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=_SIGTERM_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            _terminate(proc, logger, "keyboard interrupt")
        exit_code = 130
    finally:
        _write_status(
            status_path,
            pid=proc.pid,
            step=last_step,
            rss_mb=_tree_rss_mb(ps_proc) if proc.poll() is None else 0.0,
            best_reward=best_reward,
            plateau_count=plateau_count,
            started_at=started_at,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
