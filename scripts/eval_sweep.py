"""Evaluate a trained SAC sweep run vs TWAP on the three eval seeds.

Usage: python3 eval_sweep.py <model_dir> <out_json>

<model_dir> must contain sac.zip + vecnormalize.pkl (saved by train_sac.py
with --model-path <model_dir>/sac.zip). Obs are normalized manually with the
frozen obs_rms (no DummyVecEnv auto-reset — the terminal step's reward must
be attributed to the episode, not swallowed by an auto reset).

IMPORTANT: the currently built oep_env .so must be the SAME kappa the model
was trained on, or agent/TWAP rewards are not comparable.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/summa/optimal-execution-platform")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "build"))

from stable_baselines3 import SAC  # noqa: E402

from env.execution_env import ExecutionEnv  # noqa: E402

S0, Q, T = 100, 1000, 32
SEEDS = {"0xC0FFEEBABE": 0xC0FFEEBABE, "0xDEADBEEF": 0xDEADBEEF, "0x1234": 0x1234}


def run_episode(env, seed, act_fn):
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    trace = {"remaining": [], "size": [], "aggr": [], "reward": [],
             "cum_fills": [], "step_avg_px": []}
    prev_fills, prev_cash = 0, 0.0
    t = 0
    while not done:
        action = np.asarray(act_fn(t, obs), dtype=np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        fills = env._sim.qty_filled()
        cash = env._sim.avg_fill_price() * fills
        d_fills = fills - prev_fills
        d_cash = cash - prev_cash
        trace["remaining"].append(env._sim.remaining() / Q)
        trace["size"].append(float(action[0]))
        trace["aggr"].append(float(action[1]))
        trace["reward"].append(float(reward))
        trace["cum_fills"].append(int(fills))
        trace["step_avg_px"].append(d_cash / d_fills if d_fills > 0 else None)
        prev_fills, prev_cash = fills, cash
        t += 1
    fills = env._sim.qty_filled()
    avg_px = env._sim.avg_fill_price()
    remaining = int(env._sim.remaining())
    # Mark-to-market IS: unlike the fills-only `is_bps` (which implicitly
    # marks residual inventory at S0, i.e. zero opportunity cost), the MTM
    # variant marks the unexecuted tail at the TERMINAL mid so the agent
    # pays the true opportunity cost of resting instead of completing.
    # `obs` here is the raw ExecutionEnv observation (normalization happens
    # only inside the agent's policy wrapper), so obs[5] is the raw mid.
    final_mid = float(obs[5])
    cash_realized = fills * avg_px  # avg_px = cash/fills, so this is exact
    mtm_value = cash_realized + remaining * final_mid
    mtm_shortfall = (Q * S0) - mtm_value
    mtm_is_bps = mtm_shortfall / (Q * S0) * 1e4
    return {
        "total_reward": round(total_reward, 2),
        "fills": int(fills),
        "remaining": remaining,
        "avg_fill_px": round(avg_px, 4) if fills else None,
        "is_bps": round((S0 - avg_px) / S0 * 1e4, 1) if fills else None,
        "final_mid": round(final_mid, 4),
        "mtm_is_bps": round(mtm_is_bps, 1),
        "trace": trace,
    }


def main():
    model_dir = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    model = SAC.load(str(model_dir / "sac.zip"), device="cpu")
    with open(model_dir / "vecnormalize.pkl", "rb") as f:
        vecnorm = pickle.load(f)
    vecnorm.training = False

    def agent_act(t, obs):
        norm_obs = vecnorm.normalize_obs(obs.reshape(1, -1).astype(np.float32))
        action, _ = model.predict(norm_obs, deterministic=True)
        return action.reshape(-1)

    def twap_act(t, obs):
        return (1.0 / (T - t), 1.0)

    env = ExecutionEnv()
    results = {}
    for name, seed in SEEDS.items():
        agent = run_episode(env, seed, agent_act)
        twap = run_episode(env, seed, twap_act)
        results[name] = {"agent": agent, "twap": twap}
        print(f"[{name}] agent: r={agent['total_reward']:9.1f} "
              f"fills={agent['fills']:4d} avg={agent['avg_fill_px']} "
              f"mtm={agent['mtm_is_bps']:6.1f}bps "
              f"| twap: r={twap['total_reward']:9.1f} "
              f"fills={twap['fills']:4d} avg={twap['avg_fill_px']} "
              f"mtm={twap['mtm_is_bps']:6.1f}bps")
    env.close()

    out_json.write_text(json.dumps(results, indent=1))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
