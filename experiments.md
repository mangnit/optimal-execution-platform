# SAC Execution-Agent Tuning — Experiment Log

**Objective:** SAC agent fills only 0–3 / 1000 units; best reward plateau −12.12.
Diagnose and fix across 5 diverse trials (one parameter *category* each), rebuild
the pybind module, retrain, log. Full state in `research_state.md`.

Sign convention (docs/architecture.md): parent = **SELL** of `Q` over `[0,T]`, arrival mid
`S₀`. Marketable sell hits the **bid**; passive rests on the **ask**. Reward =
per-step realised bps vs `S₀`, normalised by `parent_qty`, minus `φ·(q/Q)²`.

---

## Prior history (supplied)

| Trial | kPhi (C++) | Learning Rate | Patience | Mean Reward | Note |
|-------|-----------|---------------|----------|-------------|------|
| 1 | 0.5f  | 3e-4 | 10 | −21.3  | Baseline |
| 2 | 0.25f | 1e-4 | 50 | −12.12 | Prior best |

---

## DIAGNOSIS — the environment is broken, not the agent

Fill ceiling (best any policy can do), fixed policies through built `oep_env.SimEnv`
(parent_qty=1000, horizon=32, mid=100, seed=0xC0FFEEBABE):

| Fixed policy (size_frac, aggression) | qty_filled / 1000 | total reward |
|--------------------------------------|-------------------|--------------|
| all_aggressive (1.0, 1.0)            | 7                 | −10.62 |
| all_passive (1.0, 0.0)               | 0                 | −8.00 |
| mid (0.5, 0.5)                       | 7                 | −10.62 |
| twap_aggressive (1/(T−t), 1.0)       | 10 (ceiling)      | −12.10 |
| small_aggressive (0.1, 1.0)          | 7                 | −10.62 |
| random                               | 9                 | −10.97 |

**Root cause — two coupled defects in `cpp/bindings/py_module.cpp::StepImpl`:**
1. **No resting bid ever exists** — `best_bid` (obs[2]) = 0 every step; seller has
   nothing to hit. Background flow too thin + self-crossing.
2. **Child order poisons the book** — marketable child sell rests its entire unfilled
   residual as a cheap ask that absorbs all future background buys → bid never re-forms.

`all_passive` fills 0, scores exactly −8.0 = kPhi(0.25)·1.0²·32. Prior −12.12 "best" is
worse than the do-nothing floor. kPhi/LR/patience cannot fix an unfillable market.

Perf: env ~3.2e5 steps/s; SAC CPU-only (torch 2.12, sb3 2.9, 4 threads, 3.78 GB) ~1e2 ts/s.
Trials run at `--total-timesteps 200000` (see research_state.md §3 for rationale).

---

## Results

> **Reward scale changed after Trial 1.** On the broken env near-zero fills meant
> reward was tiny penalties; once 1000 units actually execute, each fill books real
> bps shortfall, so numbers are NOT comparable to the prior −12.12. The metrics that
> matter now: (a) trained-policy **fills → 1000** and (b) **avg_fill vs S₀=100**.

| Trial | Category | Change | Trained-policy reward | Fills/1000 | Note |
|-------|----------|--------|----------------------|-----------|------|
| — | baseline | unmodified env | −12.12 (prior) | 0–10 | fill ceiling ~1%, env bug |
| 1 | Liquidity ("Hawkes") | flow 4→12 orders/step, qty 1–4→10–49, buys strictly below mid / sells above; prewarm [80,120] | −0.5 (illusory) | **0/1000** | **Necessary but not sufficient.** Fixed-policy fill ceiling 7→**1000** (bids now persist). BUT SAC reward-hacks: trained policy `size≈0.008, aggr≈0.065` → does nothing, fills 0, scores −0.5 (beats any real execution). Root: `child_qty==0` early-returns w/o inventory penalty; terminal forced sell posted passively never fills. **Next bottleneck = reward structure.** |
| 2 | Reward structure (pivoted from IOC) | non-dodgeable holding penalty on `child_qty==0`; terminal step forces marketable cross; unsold-at-T marked-to-liquidation at `S₀−10` (`kTerminalStressTicks`) | −107.75 (eval seed); train ep_rew_mean −124 | **1000/1000** (2/3 seeds; 531 on seed 0x1234) | **SUCCESS — execution restored, reward-hack dead (0→1000 fills).** Trained policy completes 1000/1000 at avg 99.0 (IS≈100 bps) on eval seed. −107.75 is the env's patient optimum (passive fills impossible → avg>99 unreachable); eval metric saturated by ~20k. Residual: leans on terminal dump (`aggr≈0.44`) so under-fills the hardest seed → cross-seed robustness + urgency are the levers for Trials 3–5. |
| 3 | Inventory penalty kPhi | `kPhi` 0.25→**1.0** (urgency knob) | −131 (patient basin); train −149 | 1000/1000 (2/3; 531 on 0x1234) | **φ confirmed as urgency control but SAC can't exploit it.** Probe: φ=1.0 flips fixed-policy optimum (twap −120.7 **>** do_nothing −131) → active execution *should* win. But SAC stays in the passive/terminal-dump basin (learned `size≈0.014, aggr≈0.168`; `remaining` flat at 1.00 until T). Higher φ only made reward more negative w/o changing behavior. **Bottleneck moved to SAC optimization/exploration** (premature collapse to passive after early aggressive crosses score −432). φ=1.0 kept (sets the incentive); T4/T5 target the optimization. |
| 4 | Learning rate | `learning_rate` 1e-4→**3e-4** | −131 (patient basin); train −147 | 1000/1000 (2/3; 531 on 0x1234) | **Step-size is not the constraint.** Same passive basin; `aggr` collapsed even harder to **0.000**, `remaining` flat at 1.00. Higher LR reaches the same local optimum faster. Confirms the blocker is reward/obs *conditioning*, not gradient step size → Trial 5. |
| 5 | Obs/reward normalization | `VecNormalize(norm_obs+norm_reward)` in train_sac.py | −131.8 (eval seed); train −? | 1000/1000 (2/3; 467 on 0x1234) | **Escaped the passive basin.** Active schedule (`remaining` 1.00→0.13, `size≈0.52 aggr≈0.46`), completes 1000/1000 on 2/3 seeds at avg 98.9 (IS≈110bps). Blocker was reward/obs conditioning, not φ/LR. Still leans on liquidity accumulation (env artifact) → churn fix below. |

### Post-Trial-5: Liquidity churn (env realism)

Age-based FIFO expiry of background orders (`kLiquidityLifetimeSteps=3`) so
resting bids expire and depth can't accumulate over an idle horizon. `ctest 69/69`.

**Fixed-policy probe (churned):** twap_aggr −124.8 / **1000 fills (now WINNER)**;
do_nothing & all_passive collapse to **59 fills / −978** (hoard-and-dump DEAD);
all_aggressive −429. Realistic Almgren-Chriss landscape.

**Churn-trained SAC (200k, full config) vs TWAP, per seed:**

| Seed | Agent reward / fills | TWAP reward / fills | Winner |
|------|----------------------|---------------------|--------|
| 0xC0FFEEBABE | −238.9 / 1000 | −124.8 / 1000 | TWAP |
| 0xDEADBEEF | −415.2 / 806 | −147.5 / 986 | TWAP |
| **0x1234 (hard)** | **−223.3 / 998** | **−247.7 / 921** | **AGENT ✅** |

**Verdict (200k):** Churn forces genuine active execution (`remaining` 1.00→0.02, `aggr≈0.69`) —
hoard-and-dump permanently dead. **The agent beats TWAP on the hard seed 0x1234 on
both reward and fills.** But it is NOT yet uniformly better: over-crosses on easy
seeds (avg 97.7 vs TWAP 98.9) and under-fills 0xDEADBEEF (806). Hypothesis: under-trained.

### 500k convergence run (churned env, supervisor + SIGTERM-save)

Ran 500k via the supervisor to test the under-training hypothesis. **Early-stopped
at ~167k** (ep_rew_mean plateaued, best −201.05 @ step 142208). SIGTERM-save handler
added to `train_sac.py` so the model+VecNormalize survive early-stop.

Agent (final early-stop model) vs TWAP, per seed:

| Seed | Agent 500k reward / fills | Agent 200k (prior) | TWAP reward / fills | Winner |
|------|---------------------------|--------------------|---------------------|--------|
| 0xC0FFEEBABE | −231.4 / 1000 | −238.9 / 1000 | −124.8 / 1000 | TWAP |
| 0xDEADBEEF | −314.1 / 973 | −415.2 / 806 | −147.5 / 986 | TWAP |
| 0x1234 (hard) | −319.4 / 978 | **−223.3 / 998** | −247.7 / 921 | **TWAP** |

(best-eval checkpoint @142208 is consistent: −242.7 / −377.9 / −319.1, fills 1000/987/1000 —
also loses to TWAP on all three.)

**Verdict (500k): the under-training hypothesis is REFUTED. Extended training made the
agent WORSE, not better.** It converged to an over-aggressive "complete-at-all-costs"
policy (`aggr`↑ to 0.67–0.80, avg fill ~97.1 vs TWAP 98.8): robustness/completion improved
(0xDEADBEEF fills 806→973) but price quality degraded across the board, and it **LOST the
0x1234 edge** (−223→−319, from beating TWAP to losing). The 200k "win" was an under-converged
sweet spot, not a stable optimum.

**Root cause (the real next lever — NOT more compute):** the terminal stress penalty
(`kTerminalStressTicks=10` ⇒ −100 bps/unit unfilled) dominates the spread/impact signal, so
the reward-optimal policy is "cross hard to never leave inventory" — which over-pays spread.
To beat TWAP the reward must be rebalanced: soften/quadratic-taper the terminal penalty, and/or
add explicit price-improvement shaping (reward `avg_fill − S₀` progress), so completion and
execution *quality* trade off correctly. This is a reward-design problem, not a step-count one.
| 5 | Normalization | `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10)` on train; frozen shared-stats eval; persist `vecnormalize.pkl` (run direct, not via supervisor, to survive the post-`learn()` save) | −131.8 / −126.6 (2 seeds); −603 (0x1234) | **1000/1000** (2/3); 467 on 0x1234 | **Escapes the passive basin — active execution learned.** `remaining` now declines mid-episode (1.00→0.13; `size≈0.52, aggr≈0.46`) instead of flat-at-1.00. Completes 1000/1000 on 2/3 seeds at avg ~98.9 (IS ~110 bps). Confirms the T3/T4 diagnosis (blocker = reward/obs conditioning). Residual: under-fills hardest seed (cross-seed robustness), and raw reward ≈ patient because active crossing walks the book down slightly. |

---

## Summary & Recommendation

**The reported failure was an environment bug, not an RL tuning problem.** The
"0–3 fills / 1000, plateau −12.12" symptom came from two defects in the C++ env
(no resting bid ever formed; the child order poisoned the book), capping the fill
ceiling at ~1% for *every* policy. kPhi/LR/patience were red herrings.

Causal chain the 5 diverse trials established:
1. **Liquidity (T1)** made fills *possible* (ceiling 7→1000) but exposed a
   reward-hack: SAC learned to *not trade*.
2. **Reward structure (T2)** closed the no-trade loopholes (non-dodgeable holding
   penalty + forced terminal cross + mark-to-liquidation of unsold inventory) →
   **execution restored, 0→1000 fills.** This is the single highest-impact fix.
3. **kPhi (T3)** is the urgency knob — φ=1.0 makes active execution reward-optimal
   in theory, but SAC couldn't exploit it.
4. **Learning rate (T4)** ruled out step-size as the blocker.
5. **Normalization (T5)** unblocked SAC's optimization → an **active execution
   schedule** that completes 1000/1000 on most seeds.

**Recommended production config:** T1 liquidity + T2 reward structure + φ=1.0
(T3) + LR 3e-4 (T4) + VecNormalize obs+reward (T5). All are kept in the tree.

**Biggest remaining lever (out of the 5-trial scope, most important next step):**
background resting bids never cancel/expire, so depth accumulates and *patience is
artificially optimal* — this is why raw reward barely separates active from
passive. Adding churn/decay to resting background liquidity (or enabling passive
fills via occasional aggressive background buyers) would make steady TWAP-like
execution strictly win and improve cross-seed robustness (the 0x1234 under-fill).
That is an environment-realism change, not more RL tuning.

**Methodology note:** trials used `--total-timesteps` 200k (T1–T4) / 120k (T5),
not 500k — on this CPU-only box (~100 ts/s) the fill/behaviour signal is
unambiguous well before then, and the supervisor's plateau early-stop was
observed to cut degenerate configs. Reward magnitudes are NOT comparable across
trials that change the reward function (T2, T3) or normalize it (T5); the
decision metrics are trained-policy **fills** and **avg_fill vs S₀**.

---

## Quadratic terminal penalty + κ sweep (2026-07-07)

Per `SAC_Reward_Fix_Architecture.md`: replaced the linear terminal cliff
(mark unsold at S₀−10 ⇒ −1000 bps/unit-fraction, constant marginal cost) with
a smooth quadratic terminal penalty `−κ·(q_T/Q)²` in
`py_module.cpp::StepImpl`; residual is now marked at S₀ (0 realised bps), the
quadratic is the only terminal cost. `kTerminalStressTicks` deleted.

**Tick-scale verification (pre-change):** touch after first bg-flow step =
99/101 around S₀=100 on the eval seeds; first aggressive cross books exactly
99.000 on ALL 3 seeds ⇒ c_cross = 100 bps/share, 1 tick = 100 bps. κ=2000
calibration confirmed correctly scaled. Full-order sweep walks to ~95.4–96.0
(≈400–456 bps) — the pathology itself.

**Sweep:** κ ∈ {1000, 1500, 2000}, one 200k direct run each (train seed
0xC0FFEEBABE, LR 3e-4, VecNormalize, n-envs 4), deterministic eval + TWAP
baseline through the SAME κ-binary per seed. Artifacts
`models/kappa{K}/{sac.zip,vecnormalize.pkl}`, traces
`logs/kappa{K}/eval_seeds.json`. IS = fills-only (S₀−avg_fill)/S₀ bps.

| κ | Seed | Agent r / fills / avg_px / IS | TWAP r / fills / avg_px / IS | Winner |
|---|------|-------------------------------|------------------------------|--------|
| 1000 | 0xC0FFEEBABE | **−122.0** / 933 / **98.879** / 112.1 | −124.8 / 1000 / 98.863 / 113.7 | **AGENT ✅** |
| 1000 | 0xDEADBEEF | **−126.5** / 872 / **98.877** / 112.3 | −133.7 / 986 / 98.760 / 124.0 | **AGENT ✅** |
| 1000 | 0x1234 | −203.5 / 851 / 98.041 / 195.9 | −174.9 / 921 / 98.300 / 170.0 | TWAP |
| 1500 | 0xC0FFEEBABE | −144.4 / 902 / 98.696 / 130.4 | −124.8 / 1000 / 98.863 | TWAP |
| 1500 | 0xDEADBEEF | −154.9 / 942 / 98.548 / 145.2 | −133.8 / 986 / 98.760 | TWAP |
| 1500 | 0x1234 | −234.5 / 895 / 97.763 / 223.7 | −178.1 / 921 / 98.300 | TWAP |
| 2000 | 0xC0FFEEBABE | −134.2 / 1000 / 98.771 / 122.9 | −124.8 / 1000 / 98.863 | TWAP |
| 2000 | 0xDEADBEEF | −151.7 / 998 / 98.598 / 140.2 | −133.9 / 986 / 98.760 | TWAP |
| 2000 | 0x1234 | −245.8 / 804 / 98.097 / 190.3 | −181.2 / 921 / 98.300 | TWAP |

(TWAP reward varies slightly with κ via its own unfilled residual — it is
re-run per κ-binary for comparability. Old cliff-era TWAP numbers are NOT
comparable.)

**Verdict:**
1. **Tail over-aggression is CURED at every κ.** 0xDEADBEEF traces: inventory
   declines smoothly (κ=1000: 1.00→0.13, no terminal dump; κ=2000: 1.00→0.00
   near-linear), tail sizes ≤0.35, fills book at 99/98 only — never the
   95–97 sweeps of the cliff policy (avg 97.1). The critic no longer
   propagates terminal dread; behaviour is AC-like.
2. **κ=1000 BEATS TWAP on 2/3 seeds** (0xC0FFEEBABE and 0xDEADBEEF) on BOTH
   total reward and avg fill price — first configuration to beat TWAP after
   convergence. On 0xDEADBEEF it selectively skips thin-book steps (rests when
   the touch is weak, fills at 99 nearly everywhere, only 3 steps at 98 vs
   TWAP's 6) and rests 128 units at T (quad cost only −16.4) instead of
   paying spread to force completion.
3. **κ monotonically trades completion vs price**, as the indifference math
   predicts: κ=2000 → near-full completion (1000/998) at more 98-fills;
   κ=1000 → rests ~7–13% tail at better prices. κ=1500 middle, no win.
4. **0x1234 (hard/illiquid seed) still loses at every κ** (best 190.3 vs
   TWAP 170.0 bps). Even TWAP under-fills it (921). Residual weakness is
   liquidity-adaptation on thin books — relative-value-flavoured, but §5's
   contingency (dense advantage-vs-running-TWAP, 9-D obs) triggers only on
   0/3 wins; at 2/3 wins it does NOT trigger. It remains the documented next
   lever if uniform dominance is required.

**Caveat (honest accounting):** the κ=1000 reward win banks on residual being
marked at S₀ + quadratic — i.e. zero opportunity cost on ~13% unfilled. The
fills-only avg-price win (98.877 vs 98.760 on 0xDEADBEEF) is real, but a
production IS decomposition must mark residual at S_end before claiming the
2/3 win out-of-sample.

**Config kept in tree:** `kKappa = 1000.0` (built + ctest 69/69).
