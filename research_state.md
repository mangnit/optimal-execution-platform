# research_state.md — Source of Truth (SAC Execution-Agent Fix)

> Persistent cross-session memory. Do not re-analyze the codebase; trust this file.
> Sign convention (docs/architecture.md): parent = **SELL** of `Q=1000` over `[0,T=32]`, arrival mid `S₀=100`.
> Marketable sell hits the **bid**; passive rests on the **ask**. Reward = per-step realised
> bps vs `S₀`, normalised by `parent_qty`, minus inventory penalty `φ·(q/Q)²`.

---

## 1. Current Executive Summary

The SAC agent's "0–3 fills / 1000" is **NOT** a hyperparameter problem (kPhi / LR /
patience / Hawkes are red herrings). It is two coupled bugs in
`cpp/bindings/py_module.cpp :: StepImpl` that cap the fill ceiling at ~1% for **any** policy:

1. **`best_bid` is `0` on every step of every episode.** No resting bid ever forms, so the
   seller has nothing to hit. Background flow (4 orders/step, qty 1–4, buys `[94,104]` /
   sells `[96,106]`) is too thin and mostly self-crosses.
2. **Order-book poisoning (death spiral).** A marketable child sell is priced at
   `best_bid_now` (fallback `arrival_mid−1` when no bid) and its **entire unfilled residual
   rests as a cheap ask** (`book_.AddLimit` rests the remainder, ~L285-295). That cheap ask
   absorbs every subsequent background buy, so a bid can never re-form — locking in bug (1).

Tell-tale: `all_passive` fills exactly **0** and scores exactly **−8.0 = kPhi(0.25)·1.0²·32**
(pure inventory penalty, zero fills). Prior "best" −12.12 is **worse than doing nothing** (−8.0).

Key code anchors:
- Background flow loop: `py_module.cpp` ~L215 (`for (int k = 0; k < 4; ++k)`)
- Prewarm band: `py_module.cpp` L55-56 (`kMinPrewarmPx=90`, `kMaxPrewarmPx=110`)
- Child order / residual rest: `py_module.cpp` ~L285-295
- kPhi: `py_module.cpp` L317 (`constexpr float kPhi = 0.25f;`)
- SAC hyperparams: `python/train/train_sac.py` L90-103 (`learning_rate=1e-4`, `ent_coef="auto"`)
- Supervisor early-stop: `python/train/supervisor.py` L50 (`_PLATEAU_PATIENCE = 50`)
- Obs layout (8-dim): idx0 q_rem/Q, 1 (T−t)/T, 2 best_bid, 3 best_ask, 4 spread, 5 mid,
  6 filled/Q, 7 avg_fill/S₀. **idx2-5 are RAW tick prices (~94-106) → swamp the [0,1] feats.**

---

## 1b. Progress / Outcomes (causal chain resolved)

- **T1 Liquidity** — fixed-policy fill ceiling 7→**1000**. Necessary; exposed reward-hack.
- **T2 Reward structure** — completion incentive → **execution restored (0→1000 fills)**,
  reward-hack dead. Learns patient "wait then terminal-sweep" @ avg 99 (IS≈100bps),
  −107.75 on eval seed (the env's patient optimum; passive fills impossible ⇒ avg>99 unreachable).
- **T3 kPhi 0.25→1.0** — φ confirmed as urgency knob (probe: twap −120.7 **>** do_nothing −131),
  but SAC stays in passive/terminal-dump basin (`aggr≈0.17`, remaining flat). Bottleneck → optimization.
- **T4 LR 1e-4→3e-4** — same passive basin (`aggr→0.0`). Step-size not the constraint.
- **T5 normalization** — VecNormalize(obs+reward) **escaped the passive basin**:
  active schedule (`remaining` 1.00→0.13; `size≈0.52, aggr≈0.46`), 1000/1000 on 2/3 seeds,
  avg ~98.9 (IS≈110bps); 0x1234 under-fills (467). Confirms blocker was reward/obs conditioning.
  Artifacts: `models/sac_oep_baseline.zip` + `models/vecnormalize.pkl` (needed to eval).
  Eval with `scratchpad/eval_vecnorm.py` (normalize_obs manually; NO DummyVecEnv auto-reset).
- **ALL 5 TRIALS COMPLETE. ctest 69/69 pass.** Prod config = T1+T2+φ=1.0+LR3e-4+VecNorm (all in tree).
- **Known env artifact (recommendation, out of 5-trial scope):** background resting bids
  never cancel/expire → depth accumulates → *patience is artificially optimal*. A real churn/
  decay of resting liquidity would make steady (TWAP-like) execution win and de-fang the
  terminal-dump strategy. This, not more RL tuning, is the next lever for realistic behavior.

## 1c. Liquidity churn fix (post-Trial-5, IMPLEMENTED)

`py_module.cpp`: age-based FIFO expiry of background orders. New
`kLiquidityLifetimeSteps=3`; churn FIFO members `bg_ids_`/`bg_birth_`/`bg_head_`
(reserved, allocation-free, index-dequeue); churn loop at top of `StepImpl`
(cancel born ≤ `step_idx_−L`); register (id,step) on rest; cleared in ResetInternal.
Seed-determinism preserved (ctest 69/69). **Mechanic note:** the aggressive child
sells at `best_bid_now` = TOP bid price only, so a cross takes only the top level,
not a deep sweep — churn therefore starves the accumulate-at-99 hoard.

**Probe (fixed policy, churned):** twap_aggr **−124.8 / 1000 fills (now WINNER)**;
do_nothing & all_passive collapse to **59 fills / −978** (hoard-and-dump DEAD);
all_aggressive −429. Realistic Almgren-Chriss landscape achieved.

**TWAP baseline to beat (churned env), per seed:**
| seed | TWAP reward | fills |
|------|------------|-------|
| 0xC0FFEEBABE | −124.8 | 1000 |
| 0xDEADBEEF | −147.5 | 986 |
| **0x1234 (hard)** | **−247.7** | **921** |

Even TWAP under-fills 0x1234 (921) → learned policy has room to beat it by adapting
size/aggression to book state. Training with churn: run 200k direct (vecnorm saved
at end); eval `scratchpad/eval_vecnorm.py`, baseline `scratchpad/twap_baseline.py`.

## 1d. Churn training results — 200k vs 500k (KEY: more compute ≠ better)

`train_sac.py` gained a SIGTERM/SIGINT save handler so supervisor early-stop keeps
model+vecnorm. Commits: `b47e74e` (churn). Agent-vs-TWAP reward/fills per seed:

| Seed | Agent 200k | Agent 500k (early-stop ~167k) | TWAP |
|------|-----------|-------------------------------|------|
| 0xC0FFEEBABE | −238.9 / 1000 | −231.4 / 1000 | −124.8 / 1000 |
| 0xDEADBEEF | −415.2 / 806 | −314.1 / 973 | −147.5 / 986 |
| 0x1234 (hard) | **−223.3 / 998 (beat TWAP)** | −319.4 / 978 | −247.7 / 921 |

- **200k:** beat TWAP on 0x1234 (only), lost on easy seeds (under-converged sweet spot).
- **500k:** early-stopped, best ep_rew_mean −201@142k. Converged to **over-aggressive**
  policy (`aggr` 0.67–0.80, avg fill ~97.1 vs TWAP 98.8). Robustness ↑ (0xDEADBEEF 806→973)
  but price ↓ everywhere; **LOST 0x1234 edge**. Under-training hypothesis REFUTED.
- **Root cause / next lever (reward design, NOT compute):** `kTerminalStressTicks=10`
  (−100bps/unit unfilled) dominates the spread signal → "complete-at-all-costs" over-crossing.
  Fix = soften/taper terminal penalty + add price-improvement shaping (reward `avg_fill−S₀`),
  so completion vs execution-quality trade off correctly. THEN re-train.
- Eval both artifacts: final = `models/{sac_oep_baseline.zip,vecnormalize.pkl}`;
  best-checkpoint = `logs/eval_best/best_model.zip` (+ same vecnorm). Both lose to TWAP.

**CURRENT STATE:** env is a realistic Almgren-Chriss landscape (churn live, committed).
Agent executes actively but is over-aggressive; does NOT yet beat TWAP after convergence.
Reward rebalancing is the open task. → **RESOLVED 2026-07-07, see §1e.**

## 1e. Quadratic terminal penalty — implemented + swept (2026-07-07)

Per `SAC_Reward_Fix_Architecture.md` §4.4: linear cliff (mark at S₀−10,
−1000 bps/unit-frac) REPLACED by smooth quadratic `−κ·(q_T/Q)²` in
`py_module.cpp::StepImpl`; residual marked at S₀; `kTerminalStressTicks`
deleted. Tick scale verified first: touch 99/101, first cross books 99.000 on
all 3 seeds ⇒ 1 tick = 100 bps, κ-calibration correctly scaled.

Sweep κ ∈ {1000,1500,2000} (200k direct each; κ is constexpr ⇒ rebuild per
value; TWAP re-run per κ-binary). Full table + traces: `experiments.md` and
`logs/kappa{K}/eval_seeds.json`; models `models/kappa{K}/`.

- **Over-aggression CURED at all κ** (goal of the fix): smooth inventory
  decline, no terminal dump, fills at 99/98 only (cliff policy swept 95–97).
- **κ=1000 BEATS TWAP on 2/3 seeds** on both reward and avg fill:
  0xC0FFEEBABE −122.0/933/98.879 vs TWAP −124.8/1000/98.863;
  0xDEADBEEF −126.5/872/98.877 vs TWAP −133.7/986/98.760.
  First post-convergence TWAP win. Mechanism: skips thin-book steps, rests
  ~13% tail (quad cost −16.4) instead of forcing completion at 98.
- **0x1234 (hard) still loses at all κ** (best 195.9 vs 170.0 IS bps).
  §5 contingency (advantage-vs-running-TWAP, 9-D obs) NOT triggered (needs
  0/3); it is the next lever for uniform dominance / thin-book adaptation.
- **Caveat:** reward win marks residual at S₀ (zero opportunity cost on the
  rested tail). Fills-only price win is real; mark residual at S_end in any
  production IS decomposition before claiming out-of-sample victory.
- **Tree state:** `kKappa = 1000.0` kept (winner), built, ctest 69/69.
  Milestone commit `75902d2` on `feature/sac-quadratic-terminal-k1000`.
- **MTM CORRECTION (2026-07-07, post-milestone):** `scripts/eval_sweep.py`
  (now in-repo) gained `mtm_is_bps` — residual marked at TERMINAL mid, not
  S₀. Result: **TWAP wins 3/3 on MTM IS** (agent/TWAP: 121.3/113.7,
  129.9/125.8, 203.9/168.4 bps). Terminal mid ≈ 97.5 (the forced terminal
  cross depresses the mark), so the rested tail costs ~250 bps/unit-frac —
  the κ=1000 "win" was an artifact of the S₀ residual mark. The §5
  contingency condition is met in spirit; next lever = align the in-reward
  residual mark with the terminal mid, or the §5 dense advantage-vs-TWAP
  reward (9-D obs). Cockpit shows MTM tiles alongside arrival-mid IS.

## 2. Current Baseline Metrics (fixed-policy fill ceiling)

Driven straight through built `oep_env.SimEnv` (parent_qty=1000, horizon=32, mid=100,
seed=0xC0FFEEBABE). Probe: `scratchpad/probe.py`.

| Fixed policy (size_frac, aggression) | qty_filled / 1000 | total reward |
|--------------------------------------|-------------------|--------------|
| all_aggressive (1.0, 1.0)            | 7                 | −10.62 |
| all_passive   (1.0, 0.0)             | 0                 | −8.00  |
| mid           (0.5, 0.5)             | 7                 | −10.62 |
| twap_aggr     (1/(T−t), 1.0)         | 10 (ceiling)      | −12.10 |
| small_aggr    (0.1, 1.0)             | 7                 | −10.62 |
| random                               | 9                 | −10.97 |

Prior supplied history: T1 {kPhi 0.5, LR 3e-4, patience 10} → −21.3;
T2 {kPhi 0.25, LR 1e-4, patience 50} → −12.12 (prior best). Both flailing in unfillable market.

Env perf: ~3.2e5 steps/s single-thread. SAC on this box (torch 2.12 **CPU-only**, sb3 2.9,
8 cores/4 torch-threads, **3.78 GB RAM**) is gradient-bound ~1e2 ts/s.

---

## 3. The 5-Trial Staged Plan

Forward-selection: keep each beneficial change, vary exactly ONE new category per trial,
state running config. Per-trial loop: **(a) apply edit → (b) `cmake --build build --target
oep_env` [skip for py-only T4/T5] → (c) clear `logs/` → (d) train → (e) probe fills + read
`ep_rew_mean` → (f) append row to `experiments.md`.**

**Optimization constraint:** train each trial at **`--total-timesteps 200000`** (NOT 500k).
Rationale: CPU-only ~100 ts/s → 500k≈75min×5≈6h; fill signal shows in first eval; supervisor
plateau early-stop (patience 50) cuts broken configs. Command:
`python3 -m python.train.supervisor -- --n-envs 4 --total-timesteps 200000`

### Trial 1 — Exogenous flow / liquidity ("Hawkes")  [C++, rebuild]
`py_module.cpp` background loop: `k<4`→`k<12`; qty `1+rng%4`→`10+rng%40`; price so **buys rest
strictly below mid, sells strictly above** (no self-cross): `depth=rng%5`;
`px = buy ? (mid-1-depth) : (mid+1+depth)`. Widen prewarm L55-56 to `[80,120]`.
Goal: persistent two-sided book with real bid depth.

### Trial 2 — Reward structure  [C++, rebuild]  ← PIVOTED from "IOC"
**Why pivot:** Trial 1 made fills possible but SAC reward-hacked to 0 fills
(trained policy size≈0.008, aggr≈0.065 → −0.5). Two loopholes: (a) `child_qty==0`
early-return skipped the holding penalty (dodge via size→0); (b) unfilled inventory
at T was ~free. Fix = make non-completion costly:
- namespace-scope `kPhi=0.25f`, `kTerminalStressTicks=10.0`.
- `child_qty==0` path returns `−kPhi·(q/Q)²` (no dodge).
- terminal step forces marketable cross (`aggressive = terminal || aggression>=0.5`).
- at `done_ && remaining_>0`, add stress shortfall marking unsold qty at `S₀−10`.
**Result (probe):** ALL policies now fill 1000/1000; do_nothing/all_passive −107.75,
twap −112, dump −429. Reward-hack dead; real execution gradient exists.
(Original IOC idea deferred — not needed once completion is forced.)

### Trial 3 — Inventory penalty kPhi  [C++, rebuild]
`py_module.cpp` L317: `kPhi 0.25f → 0.10f`. Re-tune φ now fills are real.

### Trial 4 — Learning rate  [Python only, no rebuild]
`train_sac.py` L102: `learning_rate 1e-4 → 3e-4` on the corrected env.

### Trial 5 — Observation normalisation  [Python only, no rebuild]
`train_sac.py`: wrap train+eval in `VecNormalize(norm_obs=True, norm_reward=False,
clip_obs=10.0)`; `eval_env.obs_rms = train_env.obs_rms`, eval `training=False`.
Raw tick-price obs (idx2-5 ~94-106) currently swamp the [0,1] features.

Full literal OLD/NEW patch text: `scratchpad/STAGED_PLAN.md`.

---

## 4. Next Immediate Steps (on execution unblock)

1. **Patch FIRST:** `cpp/bindings/py_module.cpp` — Trial 1 background-flow loop (~L215) +
   prewarm band L55-56. (Exact OLD/NEW in `scratchpad/STAGED_PLAN.md`.)
2. `cmake --build build --target oep_env`  (must succeed `-Werror`; runs `ctest` per docs/architecture.md).
3. Validate fills jumped: `python3 scratchpad/probe.py` — expect `best_bid != 0` and
   all_aggressive fills ≫ 7.
4. Clear tb dir so supervisor reads only this run: `rm -rf logs/*`.
5. Train: `python3 -m python.train.supervisor -- --n-envs 4 --total-timesteps 200000`.
6. Log Trial 1 row (kPhi, LR, patience, mean_reward, fills, notes) to `experiments.md`.
7. Proceed T2→T5 per §3, one category each.

**Env note (permission channel):** if gated ops fail with `Error: Stream closed`, the
interactive approval prompt is down — only Read/scratchpad-Write/trivial-Bash work. Retry
or have user toggle permission mode. All analysis already captured here; do not re-derive.
