"""Execution Cockpit — Streamlit dashboard for the C++ execution engine.

Drives the Pybind11-wrapped `SimEnv` via `python.env.execution_env.ExecutionEnv`
(P5.3) and, when available, the trained SAC policy from `models/`. Renders per-
step fill prices against the arrival mid and reports implementation shortfall
(bps) versus a TWAP baseline (docs/architecture.md §5, §6).

Sign convention (docs/architecture.md): parent order is a SELL of `parent_qty` units over
`horizon_steps`. IS for a SELL is measured as `(S_0 - avg_fill_price) / S_0` in
bps — positive means we underperformed the arrival mid.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_BUILD_DIR = _REPO_ROOT / "build"
_MODELS_DIR = _REPO_ROOT / "models"

for p in (str(_PYTHON_DIR), str(_BUILD_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("OEP_ENV_DIR", str(_BUILD_DIR))

from env.execution_env import ExecutionEnv  # noqa: E402


@dataclass
class StepRecord:
    step: int
    fill_qty: int
    fill_price: float
    cum_fill_qty: int
    cum_avg_price: float
    best_bid: float
    best_ask: float
    mid: float
    reward: float


@dataclass
class EpisodeResult:
    policy: str
    records: list[StepRecord]
    total_reward: float
    total_filled: int
    avg_fill_price: float
    arrival_mid: float
    parent_qty: int

    @property
    def is_bps(self) -> float:
        """Implementation shortfall in bps for a SELL parent order.

        Positive = underperformed arrival mid (worse fill). Unfilled inventory
        is marked at the arrival mid so the metric is comparable across
        episodes even when the child sweep did not empty the book.
        """
        if self.parent_qty == 0 or self.arrival_mid == 0:
            return 0.0
        proceeds = self.total_filled * self.avg_fill_price
        proceeds += (self.parent_qty - self.total_filled) * self.arrival_mid
        benchmark = self.parent_qty * self.arrival_mid
        return float((benchmark - proceeds) / benchmark * 1e4)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])


ActionFn = Callable[[np.ndarray, int, int], np.ndarray]


def twap_action(state: np.ndarray, step_idx: int, horizon: int) -> np.ndarray:
    """TWAP baseline: cross for `remaining / steps_left` every bar.

    The C++ step interprets `size_frac` as a fraction of `remaining_`, so
    `1 / (T - t)` yields uniform absolute size across the horizon. Aggression
    is pinned at 1 to guarantee the sweep lands (aggression >= 0.5 crosses).
    """
    steps_left = max(horizon - step_idx, 1)
    size_frac = 1.0 / steps_left
    return np.array([size_frac, 1.0], dtype=np.float32)


def make_sac_action_fn(model_path: Path) -> Optional[ActionFn]:
    """Load the SB3 SAC actor and return a per-step action selector."""
    try:
        from stable_baselines3 import SAC
    except Exception as exc:  # pragma: no cover
        st.warning(f"stable_baselines3 unavailable: {exc}")
        return None
    if not model_path.exists():
        return None
    model = SAC.load(str(model_path), device="cpu")

    def _act(state: np.ndarray, step_idx: int, horizon: int) -> np.ndarray:
        action, _ = model.predict(state, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return _act


def run_episode(
    policy_name: str,
    action_fn: ActionFn,
    parent_qty: int,
    horizon_steps: int,
    arrival_mid: int,
    seed: int,
) -> EpisodeResult:
    env = ExecutionEnv(
        parent_qty=parent_qty,
        horizon_steps=horizon_steps,
        arrival_mid=arrival_mid,
        seed=seed,
    )
    try:
        obs, _ = env.reset(seed=seed)
        records: list[StepRecord] = []
        total_reward = 0.0
        prev_cum_qty = 0
        prev_cum_cash = 0.0
        for step_idx in range(horizon_steps):
            action = action_fn(np.asarray(obs, dtype=np.float32), step_idx, horizon_steps)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            cum_qty = int(env._sim.qty_filled())
            cum_avg = float(env._sim.avg_fill_price())
            cum_cash = cum_qty * cum_avg

            step_qty = cum_qty - prev_cum_qty
            step_cash = cum_cash - prev_cum_cash
            step_price = (step_cash / step_qty) if step_qty > 0 else float("nan")

            records.append(
                StepRecord(
                    step=step_idx,
                    fill_qty=step_qty,
                    fill_price=step_price,
                    cum_fill_qty=cum_qty,
                    cum_avg_price=cum_avg,
                    best_bid=float(obs[2]),
                    best_ask=float(obs[3]),
                    mid=float(obs[5]),
                    reward=float(reward),
                )
            )
            prev_cum_qty = cum_qty
            prev_cum_cash = cum_cash
            if terminated or truncated:
                break

        return EpisodeResult(
            policy=policy_name,
            records=records,
            total_reward=total_reward,
            total_filled=prev_cum_qty,
            avg_fill_price=records[-1].cum_avg_price if records else 0.0,
            arrival_mid=float(arrival_mid),
            parent_qty=parent_qty,
        )
    finally:
        env.close()


def build_execution_chart(
    policy_result: EpisodeResult,
    twap_result: EpisodeResult,
    arrival_mid: float,
) -> go.Figure:
    fig = go.Figure()

    fig.add_hline(
        y=arrival_mid,
        line=dict(color="#888", dash="dash", width=1),
        annotation_text=f"arrival mid = {arrival_mid:g}",
        annotation_position="top left",
    )

    for res, color in ((policy_result, "#1f77b4"), (twap_result, "#ff7f0e")):
        df = res.to_frame()
        filled = df[df["fill_qty"] > 0]
        fig.add_trace(
            go.Scatter(
                x=filled["step"],
                y=filled["fill_price"],
                mode="lines+markers",
                name=f"{res.policy} fill price",
                line=dict(color=color, width=2),
                marker=dict(size=6 + np.clip(filled["fill_qty"] / max(res.parent_qty, 1) * 40, 0, 12)),
                hovertemplate=(
                    "step %{x}<br>fill px %{y:.3f}<br>"
                    "qty %{customdata}<extra>" + res.policy + "</extra>"
                ),
                customdata=filled["fill_qty"],
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["cum_avg_price"].replace(0.0, np.nan),
                mode="lines",
                name=f"{res.policy} cum avg",
                line=dict(color=color, dash="dot", width=1),
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Fill price vs arrival mid",
        xaxis_title="step",
        yaxis_title="price (ticks)",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=50, b=40),
        height=440,
    )
    return fig


def render() -> None:
    st.set_page_config(page_title="Execution Cockpit", layout="wide")
    st.title("Execution Cockpit")
    st.caption(
        "Drives the C++ engine through the Pybind11 bridge (P5.2/5.3). "
        "Sign: parent SELL of Q units over horizon [0, T]."
    )

    with st.sidebar:
        st.header("Parent order")
        parent_qty = st.number_input(
            "parent_qty", min_value=1, max_value=1_000_000, value=1000, step=100
        )
        horizon_steps = st.number_input(
            "horizon_steps", min_value=2, max_value=512, value=32, step=1
        )
        arrival_mid = st.number_input(
            "arrival_mid (ticks)", min_value=1, max_value=100_000, value=100, step=1
        )
        seed = st.number_input(
            # Streamlit ships bounds as JSON — JS ints max out at 2^53-1
            # (StreamlitJSNumberBoundsError otherwise). Xorshift64 doesn't
            # care; it re-seeds mod 2^64 downstream.
            "seed", min_value=0, max_value=(1 << 53) - 1,
            value=0xC0FFEEBABE, step=1, format="%d",
        )

        st.header("Policy")
        sb3_path = _MODELS_DIR / "sac_oep_baseline.zip"
        sac_available = sb3_path.exists()
        policy_options = ["TWAP", "Random"]
        if sac_available:
            policy_options.insert(0, "SAC")
        policy = st.selectbox("policy", policy_options, index=0)
        if not sac_available:
            st.caption(f"SAC unavailable — expected {sb3_path.relative_to(_REPO_ROOT)}")

        run = st.button("Run Execution", type="primary", use_container_width=True)

    if not run:
        st.info("Configure parameters in the sidebar, then hit **Run Execution**.")
        return

    action_fn: ActionFn
    if policy == "SAC":
        sac_fn = make_sac_action_fn(sb3_path)
        if sac_fn is None:
            st.error("Failed to load SAC policy. Falling back to TWAP.")
            action_fn = twap_action
            policy = "TWAP (fallback)"
        else:
            action_fn = sac_fn
    elif policy == "Random":
        rng = np.random.default_rng(int(seed))
        def _random(state, step_idx, horizon):
            return rng.random(2, dtype=np.float32)
        action_fn = _random
    else:
        action_fn = twap_action

    with st.spinner(f"Running {policy}…"):
        policy_result = run_episode(
            policy, action_fn,
            int(parent_qty), int(horizon_steps), int(arrival_mid), int(seed),
        )

    with st.spinner("Running TWAP baseline…"):
        twap_result = run_episode(
            "TWAP", twap_action,
            int(parent_qty), int(horizon_steps), int(arrival_mid), int(seed),
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{policy} IS (bps)", f"{policy_result.is_bps:+.2f}")
    c2.metric("TWAP IS (bps)", f"{twap_result.is_bps:+.2f}")
    delta = policy_result.is_bps - twap_result.is_bps
    c3.metric("Δ vs TWAP (bps)", f"{delta:+.2f}",
              delta=f"{-delta:+.2f}", delta_color="normal")
    c4.metric(
        "Filled / Parent",
        f"{policy_result.total_filled} / {policy_result.parent_qty}",
        delta=f"avg px {policy_result.avg_fill_price:.3f}",
    )

    st.plotly_chart(
        build_execution_chart(policy_result, twap_result, float(arrival_mid)),
        use_container_width=True,
    )

    with st.expander(f"{policy} step tape", expanded=False):
        st.dataframe(policy_result.to_frame(), use_container_width=True)
    with st.expander("TWAP step tape", expanded=False):
        st.dataframe(twap_result.to_frame(), use_container_width=True)


if __name__ == "__main__":
    render()
