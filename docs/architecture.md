# Optimal-Execution Platform — Final Build Blueprint

**Thesis:** a low-latency C++ execution engine whose validated product is an RL **optimal-execution
agent** that minimises **Implementation Shortfall (IS), in basis points**, against Almgren–Chriss /
TWAP / VWAP baselines. The same control core can be reparameterised into a market maker — that is
an architecture note, not a second product.

Convention used everywhere: a parent **SELL** of `Q` units over horizon `[0, T]`, arrival mid `S_0`.
One direction, one sign convention, no exceptions. IS sign bugs are the #1 silent failure.

---

## 0. The two clocks (the mental model that governs every decision)

The system runs on two independent clocks. Confusing them is the mistake the original hype-driven
version made; keeping them separate is what makes this architecture defensible.

- **Internal clock — the matching / simulation engine.** Nanosecond-deterministic, high-throughput,
  benchmarked. It exists to (a) generate statistically valid fills during heavy Hawkes bursts
  without dropping events, and (b) run millions of episodes fast enough to train RL in hours. The
  C++ micro-optimisations (slab allocator, intrusive LOB, lock-free rings, core pinning, CRTP) serve
  **this** clock.
- **External clock — the parent-order horizon and the exchange edge.** Minutes to hours. The RL
  agent's live scheduling decisions and the FIX gateway live here. Latency in the tens of
  microseconds is irrelevant on this clock; correctness, protocol coverage, and recovery matter.

Every "why did you optimise X" question resolves to: *which clock is X on?* FIX is on the external
clock, so it is **off the latency benchmark**. The matching path is on the internal clock, so it is
**on** it.

---

## 1. North-star metric: Implementation Shortfall

Every C++ optimisation and every RL epoch is judged by whether it drives IS down with a tighter
distribution. You are not claiming to print money — you are claiming to reduce friction against a
mathematically bounded benchmark. That claim survives senior scrutiny.

For a SELL, decision/arrival price `S_0`, child fills `n_j` at prices `p_j`:

```
IS_cash = Q·S_0  -  Σ_j n_j·p_j  -  (unfilled)·S_end
IS_bps  = IS_cash / (Q · S_0) · 10_000
```

Report the **decomposition**, not just the scalar: market-impact cost, timing/delay cost,
opportunity cost (unexecuted at `T`), explicit fees/spread. Report mean **and** the 95th/99th
percentile tail — execution algos live or die in the tail.

---

## 2. Architecture — threading & data flow

The headline structural rule: **the hot path writes to exactly one SPSC ring and never blocks.**
A separate relay thread does all slow fan-out. No SPMC on the hot path.

```
  INTERNAL CLOCK — isolated core (pinned, bare metal, deterministic, BENCHMARKED)
  ┌──────────────────────────────────────────────────────────────────┐
  │  UDP multicast feed  →  LOB (intrusive LevelFIFO)                  │
  │     →  strategy eval (CRTP: TWAP / AC / VWAP / RL-LibTorch)        │
  │     →  matching + queue-aware fill (sim)                          │
  │  ◄── this is the only path measured in latency_report.md ──►       │
  └───────────────────────────┬────────────────────────────────────────┘
                              │ write-only, lock-free, NEVER blocks
                              ▼
        SPSC ring  ── bounded · overflow-tolerant · std::atomic<uint64_t> dropped_messages
                              │
                              ▼
  EXTERNAL CLOCK — relay thread (non-pinned, does the slow work)
  ┌──────────────────────────────────────────────────────────────────┐
  │  → kdb+ writer (C API)                                             │
  │  → FIX gateway (QuickFIX)   ← edge protocol, slower clock, OFF-bench│
  │  → dashboard socket (Streamlit cockpit)                           │
  └──────────────────────────────────────────────────────────────────┘
```

If the relay stalls (page fault, OS hiccup, slow IO), the hot path keeps running, the ring
overwrites its oldest telemetry, and `dropped_messages` ticks up. The hot path is allowed to outrun
its observers; we drop telemetry and preserve matching latency. Telemetry loss is acceptable;
hot-path jitter is not.

---

## 3. Stack

**Internal clock (hot path) — bare metal, never containerised in benchmarks**
- C++20, CMake + Ninja
- `std::atomic` + hand-rolled **SPSC** ring (single hot path → relay); `pthread_setaffinity_np`
- Intrusive `LevelFIFO` price levels (O(1) cancel); slab/pool allocator (O(1), no hot-path `new`);
  `alignas(64)` to kill false sharing
- CRTP strategy dispatch (no virtual, no exceptions, no RTTI on the hot path)

**Relay thread + external clock**
- kdb+ Personal Edition (free, non-commercial) via C API (`c.h`), async
- QuickFIX (C++) **edge gateway**: `NewOrderSingle` out, `ExecutionReport` in, seq-num + reconnect.
  Explicitly off the hot-path benchmark.
- UDP multicast feed (`IP_ADD_MEMBERSHIP`) with sequence-gap detection + recovery; epoll TCP

**Research / ML**
- Python 3.11+, PyTorch, Stable-Baselines3, Gymnasium
- Pybind11 bridge — **vectorised envs, flat float buffers, GIL released in the C++ step** (see §6, P5)
- TorchScript (`.pt`) loaded by LibTorch for in-process C++ inference at deploy time (no crossing)

**Sim**
- C++ marked Hawkes generator (exp + power-law), Ogata thinning, subcritical guard `n<1`
- Discrete-event matching engine + queue-position-aware fill model
- **Seeded determinism**: identical seed ⇒ identical fill sequence, every run, every machine

**Viz / infra**
- Streamlit "execution cockpit" (the headline visual — §8)
- Optional, last: Qt monitor consuming a *separate* SPSC feed off the relay (never the hot path)
- Docker + docker-compose for build, kdb, training, dashboard — never the hot-path benchmark
- CI (GitHub Actions): build + GoogleTest + benchmark-regression gate
- Tooling: GoogleTest, Google Benchmark, HdrHistogram, spdlog (async), perf/VTune

---

## 4. Repository layout

```
optimal-execution-platform/
├── CMakeLists.txt
├── README.md                      # latency table + IS-vs-baseline frontier, above the fold
├── .github/workflows/ci.yml
├── docker/{Dockerfile.build, Dockerfile.kdb, Dockerfile.train, docker-compose.yml}
├── cpp/
│   ├── core/
│   │   ├── types.hpp              # POD, cache-aligned
│   │   ├── slab_allocator.hpp
│   │   ├── spsc_ring.hpp          # bounded, overflow-tolerant, drop-counter
│   │   └── order_book.{hpp,cpp}   # intrusive LevelFIFO, O(1) cancel
│   ├── relay/
│   │   └── distributor.{hpp,cpp}  # drains SPSC, fans out to kdb/FIX/dashboard
│   ├── net/{udp_multicast, tcp_server, fix_gateway}.{hpp,cpp}
│   ├── sim/{hawkes, matching_engine}.{hpp,cpp}, queue_model.hpp
│   ├── exec/
│   │   ├── strategy.hpp           # CRTP interface
│   │   ├── twap.{hpp,cpp}         # Phase-2 infra validation + benchmark
│   │   ├── vwap.{hpp,cpp}
│   │   ├── almgren_chriss.{hpp,cpp}
│   │   ├── rl_policy.{hpp,cpp}    # LibTorch forward pass -> action
│   │   └── shortfall.{hpp,cpp}    # IS + decomposition (hand-verified sign test)
│   ├── data/kdb_writer.{hpp,cpp}
│   ├── bindings/py_module.cpp     # pybind11, vectorised, GIL-released step
│   ├── app/{sim_runner, live_loop}.cpp
│   ├── bench/                     # google benchmark (internal clock ONLY)
│   └── tests/                     # gtest: book, allocator, IS sign, determinism
├── python/
│   ├── env/execution_env.py       # Gymnasium; vectorised
│   ├── train/{reward.py, train_sac.py, export_torchscript.py}
│   └── analysis/{calibrate_hawkes.py, tca.py}
├── q/{schema.q, tca.q}
├── dashboard/app.py
├── qt/                            # optional, last
├── benchmarks/                    # committed reports + flamegraphs
└── docs/{architecture.md, latency_report.md, math.md, limitations.md, positioning.md}
```

Note there is no `spmc_ring.hpp`. The fan-out lives in `relay/distributor`, fed by one SPSC ring.

---

## 5. The math

### 5.1 Almgren–Chriss (primary baseline)
Liquidate `Q` over `[0,T]`, `N` steps (`τ=T/N`), linear temporary impact `η`, vol `σ`, risk
aversion `λ`. Urgency (continuous limit):
```
κ = sqrt( λ σ² / η )
x_j = Q · sinh( κ (T - t_j) ) / sinh( κ T ) ,   n_j = x_{j-1} - x_j
```
`λ→0` ⇒ `κ→0` ⇒ linear = TWAP. Large `λ` ⇒ front-loaded. Objective: `min E[IS] + λ·Var[IS]`
(the efficient frontier). RL learns the state-dependent policy AC's static curve cannot.

### 5.2 TWAP / VWAP (benchmarks)
- TWAP: `n_j = Q/N`; benchmark = time-average mid.
- VWAP: `n_j = Q·u_j` with volume profile `Σu_j=1`; benchmark = volume-weighted traded price.

### 5.3 Hawkes (synthetic flow)
```
λ(t) = μ + Σ_{t_i<t} α · m_i · e^{-β(t - t_i)}
```
Branching ratio `n=α/β`; require `n<1`. Ogata thinning. Power-law `φ(u)=α(u+c)^{-γ}` for long
memory. **Calibrate to real ticks** (KS test, rescaled QQ) — §9.

### 5.4 Execution RL — state / action / reward
State (normalised): `q_t/Q`, `(T-t)/T`, spread, OFI, microprice−mid, rolling volatility, queue
position, recent fill ratio.
Action (continuous): `[size_fraction∈[0,1], aggression∈[0,1]]` (post / join / cross).
Reward (SELL — capture `S_0`, finish by `T`):
```
r_t  =  n_t·(p_t - S_0)/S_0·10_000   -   φ·(q_t/Q)²
r_T  +=  forced liquidation of q_T at market sweep   (+ optional -α·(q_T/Q)²)
```
Maximising `Σ r_t` ≡ minimising `E[IS] + (risk-aversion)·Var[IS]` — the AC objective. Sweep `φ` to
trace an empirical efficient frontier vs the analytic AC frontier (the cockpit's hero chart).
Algorithm: SAC primary (continuous, off-policy, entropy exploration); PPO as a stability check.

### 5.5 Queue-position fill model
A resting order fills only once cumulative executed volume at its level since posting reaches the
queue-ahead `Q_ahead`; cancellations ahead reduce it probabilistically. Ignoring this inflates
passive profitability and is the first thing a reviewer probes.

---

## 6. Phased roadmap + timeline

Assumes ~12 hrs/week part-time. Full-time compresses roughly 3×. RL convergence (P5) is the wildcard
— budget slack there. **Minimum shippable interview project = end of Phase 3** (a complete HFT C++
systems project, no RL). Phases 4–6 add the research differentiator.

| Phase | Weeks | Cum. | Gate ("done when") |
|---|---|---|---|
| P0 Scaffolding | 1 | wk 1 | `docker build` + `cmake --build` + `ctest` pass in CI |
| P1 C++ core | 4 | wk 5 | `latency_report.md` with p50/p99/p99.9; zero hot-path alloc verified; `perf` cache/branch-miss numbers |
| P2 Sim + TWAP + determinism | 4 | wk 9 | realistic fill/IS tape under TWAP; **same seed ⇒ identical fills**; Hawkes KS test passes |
| P3 Networking | 3 | wk 12 | multicast ingest + FIX round-trip; forced-disconnect → clean seq recovery |
| P4 Data + TCA | 3 | wk 15 | `q/tca.q` produces full IS-bps decomposition from stored ticks |
| P5 Baselines + RL | 7 | wk 22 | RL beats TWAP and matches/beats AC on **out-of-sample** mean IS with tighter tail; C++ inference latency reported |
| P6 Cockpit + packaging | 4 | wk 26 | `docker compose up` brings up analytics stack; README sells it in 90s |

**Per-phase notes**
- **P1** is the SDE deliverable — do not rush it to reach RL. Slab allocator, SPSC ring
  (overflow-tolerant + drop counter), intrusive LOB, benchmark harness, thread pinning.
- **P2** introduces the relay thread and a hardcoded TWAP to battle-test the core under load before
  any RL. Lock in seeded determinism here — it's a test, not a hope.
- **P3** FIX is wired through the relay, on the external clock. It does **not** enter `bench/`.
- **P5** the throughput trap: use SB3 vectorised envs, marshal flat float arrays across Pybind11,
  release the GIL in the C++ step. Run whole episodes in C++ where possible. Implement AC + VWAP in
  C++ first as benchmarks, then train SAC, then export TorchScript and run inference in `rl_policy`.
- **P6** Qt only if time remains, and only as a relay consumer (zero-copy IPC framing), never the
  hot path.

---

## 7. Benchmarking scope (what is and isn't measured)

`latency_report.md` measures **only the internal clock**:
```
Synthetic order arrival → LOB update → strategy evaluation  (and matching/fill in sim)
```
Explicitly **excluded** from the nanosecond benchmark: FIX serialisation, kdb writes, dashboard IO —
all on the relay / external clock by design.

Do not pre-commit to marketing numbers ("sub-100ns"). Measure, commit the HdrHistogram, and let the
distribution speak. Target a tight, low-microsecond (or better) book-update + eval path in sim, and
report the tail honestly.

---

## 8. The execution cockpit (highest-leverage visual)

One Streamlit screen, built to look like a desk tool:
- Benchmark trajectory (VWAP / AC `sinh` schedule) as a smooth line.
- The agent's actual child placements overlaid, sized by quantity, coloured by aggression.
- Live IS-bps readout: agent vs each baseline, mean **and** worst-case tail.
- Remaining-inventory curve vs the AC `sinh` curve.
- **Hero chart:** empirical efficient frontier (sweep `φ`) over the analytic AC frontier. If the
  learned frontier sits at or inside AC's, you've shown the agent exploits microstructure (OFI,
  microprice decay) a closed-form equation can't see.

---

## 9. Validation protocol (inoculates against "graded your own homework")

In `docs/limitations.md`, linked from the README:
- Calibrate the Hawkes sim to real ticks (KS test + rescaled QQ); report the fit.
- Train and evaluate on **disjoint seed sets**; report out-of-sample IS only.
- State the sim's known unrealisms (no adversarial latency arbitrage, simplified cancels,
  single-venue) and that results are therefore upper bounds.
- Report distributions, not point estimates.
- Never quote raw dollar profit. The deliverable is reduced friction vs a named benchmark.

---

## 10. Positioning — how to talk about it (`docs/positioning.md`)

**The SPSC drop-counter (systems maturity):** "The hot path is strictly bounded and allowed to
outrun its observers; if downstream IO stalls, we drop telemetry, increment a drop counter, and
preserve matching latency."

**FIX framing (domain awareness):** "QuickFIX is the industry-standard agency edge gateway. It lives
on the external clock — the parent-order horizon — so it's deliberately outside the nanosecond
benchmark. A prop HFT setup would swap it for a native binary protocol like OUCH or CME iLink; that's
a different problem domain, not a fix for this one."

**The honesty of the microsecond (the two-pronged answer to "why nanoseconds for a 30-minute
algo?"):**
1. *Engineering reality:* "The micro-optimisations aren't for the live decision loop; they maximise
   Hawkes-simulator throughput so the RL model trains in hours, not days."
2. *Brutal honesty:* "And I built this to prove I can engineer deterministic, bare-metal low-latency
   C++ — the infrastructure this role requires. I know a production VWAP algo doesn't need
   nanosecond tick-to-trade, and I wouldn't waste this effort on its routing path."

Owning the portfolio context beats inventing a fake technical justification.

---

## 11. Market-making swap (one paragraph, as planned)

The control core is strategy-agnostic. A rolling/zero terminal-inventory target, two-sided quoting,
and a moderate running-inventory penalty `φ` reparameterise the same objective into market making
(Avellaneda–Stoikov as its closed-form baseline) — both are parameterisations of one
stochastic-control problem (Cartea–Jaimungal–Penalva). Shipped as a small proof-of-concept config,
not a second validated product. The execution head is the product; the swap proves the architecture
generalises.

---

## 12. Testing & determinism

- **Unit:** order book (insert/cancel/match invariants), slab allocator (O(1), no leak), SPSC ring
  (overflow correctness + drop count).
- **IS sign test:** a hand-computed parent order whose IS you know on paper; assert exact match.
  Write this before trusting any aggregate.
- **Determinism test:** fixed seed ⇒ byte-identical fill sequence across runs and machines.
- **Benchmark regression:** CI fails if p99 of the internal path regresses past a threshold.
- **Property/fuzz** the order book with random valid op sequences.

---

## 13. Risk register (what kills this, and the mitigation)

| Risk | Mitigation |
|---|---|
| RL never converges | Ship at end of P3 (systems project stands alone); bound RL scope; SAC + simple state first |
| Pybind11 per-step overhead dominates training | Vectorised envs, flat buffers, GIL released, episodes in C++ |
| Sim non-determinism makes RL undebuggable | Seeded determinism as a P2 test gate |
| IS computed with wrong sign | Hand-verified sign unit test before any aggregation |
| Scope creep into both heads / FPGA | Execution head only; MM is one paragraph; no FPGA |
| Latency claims overreach | Measure and commit histograms; no marketing numbers |

---

## 14. Definition of done (README above the fold)

1. Architecture diagram (the two-clock data flow).
2. Latency table (internal clock, p50/p99/p99.9, with `perf` counters).
3. The efficient-frontier hero chart: learned frontier vs analytic Almgren–Chriss.
4. A one-line honest results statement: IS reduced by N bps vs TWAP / vs AC, out-of-sample, with
   tail figures and a link to `limitations.md`.
