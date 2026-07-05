// Unit tests for the Almgren-Chriss baseline (task P5).
//
// What we prove:
//   1. κ = sqrt(λ σ² / η) is what the ctor stores.
//   2. Boundary anchors: remaining_[0] == Q, remaining_[N] == 0, Σ child == Q.
//   3. Schedule is monotone non-increasing in `remaining`.
//   4. λ = 0 collapses to TWAP (uniform child sizes).
//   5. Positive λ front-loads: mid-way remaining < TWAP mid-way remaining.
//   6. Hand-computed check at N=4 with a specific (λ, σ, η, T, Q).
//   7. Zero heap allocation during ChildQuantity/RemainingAfterStep on the hot
//      path (per-step lookup does not touch the global allocator).
//   8. SubmitStep pushes an Ack event through TelemetryPublisher into the
//      SPSC ring for a passive (non-crossing) child order.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "exec/almgren_chriss.hpp"
#include "exec/telemetry_publisher.hpp"
#include "external/kdb_logger.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

// Intercept every operator new/delete variant so we can prove the hot-path
// step is allocation-free. Same shape as system_integration_test.cpp — kept
// self-contained so this TU can be built independently.
namespace {
std::atomic<std::size_t> g_new_calls{0};
std::atomic<std::size_t> g_del_calls{0};
std::size_t TotalNews() { return g_new_calls.load(std::memory_order_relaxed); }
std::size_t TotalDels() { return g_del_calls.load(std::memory_order_relaxed); }

void* DoAlloc(std::size_t sz, std::size_t align) {
  g_new_calls.fetch_add(1, std::memory_order_relaxed);
  void* p = nullptr;
  if (align < sizeof(void*)) align = sizeof(void*);
  if (::posix_memalign(&p, align, sz == 0 ? 1 : sz) != 0) {
    throw std::bad_alloc();
  }
  return p;
}

void DoFree(void* p) {
  if (p == nullptr) return;
  g_del_calls.fetch_add(1, std::memory_order_relaxed);
  std::free(p);
}
}  // namespace

void* operator new(std::size_t sz) { return DoAlloc(sz, alignof(std::max_align_t)); }
void* operator new[](std::size_t sz) { return DoAlloc(sz, alignof(std::max_align_t)); }
void* operator new(std::size_t sz, std::align_val_t al) {
  return DoAlloc(sz, static_cast<std::size_t>(al));
}
void* operator new[](std::size_t sz, std::align_val_t al) {
  return DoAlloc(sz, static_cast<std::size_t>(al));
}
void operator delete(void* p) noexcept { DoFree(p); }
void operator delete[](void* p) noexcept { DoFree(p); }
void operator delete(void* p, std::size_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::size_t) noexcept { DoFree(p); }
void operator delete(void* p, std::align_val_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::align_val_t) noexcept { DoFree(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { DoFree(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { DoFree(p); }

namespace {

using oep::core::Quantity;
using oep::core::Side;
using oep::exec::AlmgrenChriss;

TEST(AlmgrenChriss, KappaFormula) {
  EXPECT_DOUBLE_EQ(AlmgrenChriss::ComputeKappa(0.0, 1.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(AlmgrenChriss::ComputeKappa(1.0, 0.0, 1.0), 0.0);
  // λ σ² / η = 4 · 3² / 9 = 4 → κ = 2.
  EXPECT_DOUBLE_EQ(AlmgrenChriss::ComputeKappa(4.0, 3.0, 9.0), 2.0);
}

TEST(AlmgrenChriss, BoundaryAnchorsAndTotal) {
  AlmgrenChriss ac({/*parent_qty=*/1000,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/10,
                    /*horizon=*/1.0,
                    /*lambda=*/1e-4,
                    /*sigma=*/0.3,
                    /*eta=*/1e-6});
  EXPECT_EQ(ac.RemainingAfterStep(0), 1000u);
  EXPECT_EQ(ac.RemainingAfterStep(ac.num_steps()), 0u);
  Quantity sum = 0;
  for (std::size_t j = 0; j < ac.num_steps(); ++j) {
    sum += ac.ChildQuantity(j);
  }
  EXPECT_EQ(sum, ac.parent_qty());
}

TEST(AlmgrenChriss, RemainingIsMonotoneNonIncreasing) {
  AlmgrenChriss ac({/*parent_qty=*/500,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/25,
                    /*horizon=*/60.0,
                    /*lambda=*/1e-3,
                    /*sigma=*/0.5,
                    /*eta=*/1e-5});
  for (std::size_t j = 1; j <= ac.num_steps(); ++j) {
    EXPECT_LE(ac.RemainingAfterStep(j), ac.RemainingAfterStep(j - 1))
        << "at j=" << j;
  }
}

TEST(AlmgrenChriss, LambdaZeroCollapsesToTwap) {
  // λ = 0 ⇒ κ = 0 ⇒ linear schedule ⇒ equal-sized children (TWAP).
  AlmgrenChriss ac({/*parent_qty=*/100,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/10,
                    /*horizon=*/1.0,
                    /*lambda=*/0.0,
                    /*sigma=*/0.5,
                    /*eta=*/1e-5});
  EXPECT_DOUBLE_EQ(ac.kappa(), 0.0);
  for (std::size_t j = 0; j < ac.num_steps(); ++j) {
    EXPECT_EQ(ac.ChildQuantity(j), 10u) << "at j=" << j;
  }
}

TEST(AlmgrenChriss, LargeLambdaIsFrontLoaded) {
  const Quantity Q = 1000;
  const std::size_t N = 20;
  AlmgrenChriss twap({Q, Side::kSell, N, 1.0, 0.0, 0.5, 1e-5});
  AlmgrenChriss urgent({Q, Side::kSell, N, 1.0, 1.0, 0.5, 1e-6});
  // At the mid-point, the urgent schedule should have already worked more of
  // the parent than the linear TWAP schedule.
  EXPECT_LT(urgent.RemainingAfterStep(N / 2), twap.RemainingAfterStep(N / 2));
}

TEST(AlmgrenChriss, HandComputedSmallSchedule) {
  // N=4, T=1, κ=2 ⇒ κT=2. Continuous limits:
  //   x_0 = Q = 1000
  //   x_1 = 1000·sinh(1.5)/sinh(2) = 1000·2.1293/3.6269 ≈ 587.11
  //   x_2 = 1000·sinh(1.0)/sinh(2) = 1000·1.1752/3.6269 ≈ 324.03
  //   x_3 = 1000·sinh(0.5)/sinh(2) = 1000·0.5211/3.6269 ≈ 143.68
  //   x_4 = 0
  // Rounded remaining table: [1000, 587, 324, 144, 0]
  // Child sizes:              [413, 263, 180, 144]  (Σ = 1000)
  AlmgrenChriss ac({/*parent_qty=*/1000,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/4,
                    /*horizon=*/1.0,
                    // Pick λ, σ, η so ComputeKappa returns exactly 2.
                    /*lambda=*/4.0,
                    /*sigma=*/3.0,
                    /*eta=*/9.0});
  EXPECT_DOUBLE_EQ(ac.kappa(), 2.0);
  EXPECT_EQ(ac.RemainingAfterStep(0), 1000u);
  EXPECT_EQ(ac.RemainingAfterStep(1), 587u);
  EXPECT_EQ(ac.RemainingAfterStep(2), 324u);
  EXPECT_EQ(ac.RemainingAfterStep(3), 144u);
  EXPECT_EQ(ac.RemainingAfterStep(4), 0u);
  EXPECT_EQ(ac.ChildQuantity(0), 413u);
  EXPECT_EQ(ac.ChildQuantity(1), 263u);
  EXPECT_EQ(ac.ChildQuantity(2), 180u);
  EXPECT_EQ(ac.ChildQuantity(3), 144u);
}

TEST(AlmgrenChriss, HotPathHasNoHeapTraffic) {
  AlmgrenChriss ac({/*parent_qty=*/10'000,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/128,
                    /*horizon=*/1.0,
                    /*lambda=*/1e-3,
                    /*sigma=*/0.4,
                    /*eta=*/1e-5});
  // Warm-up read so any lazy static init in the test harness allocates before
  // we snapshot the counters.
  volatile Quantity sink = 0;
  for (std::size_t j = 0; j < ac.num_steps(); ++j) {
    sink += ac.ChildQuantity(j);
    sink += ac.RemainingAfterStep(j);
  }

  const std::size_t n0 = TotalNews();
  const std::size_t d0 = TotalDels();
  for (int rep = 0; rep < 1000; ++rep) {
    for (std::size_t j = 0; j < ac.num_steps(); ++j) {
      sink += ac.ChildQuantity(j);
      sink += ac.RemainingAfterStep(j);
    }
  }
  EXPECT_EQ(TotalNews(), n0);
  EXPECT_EQ(TotalDels(), d0);
  (void)sink;
}

TEST(AlmgrenChriss, SubmitStepEmitsAckThroughPublisher) {
  using Event = oep::external::TelemetryEvent;
  using Ring = oep::core::SpscRing<Event, 64>;
  oep::core::OrderBook book(64);
  // Prewarm the ask price so a passive SELL child never triggers a map
  // node allocation.
  book.PrewarmLevel(Side::kSell, 105);
  Ring ring;
  auto clock = []() noexcept -> std::uint64_t { return 42; };
  oep::exec::TelemetryPublisher publisher(book, ring, clock);

  AlmgrenChriss ac({/*parent_qty=*/50,
                    /*parent_side=*/Side::kSell,
                    /*num_steps=*/5,
                    /*horizon=*/1.0,
                    /*lambda=*/0.0,
                    /*sigma=*/0.5,
                    /*eta=*/1e-5});
  ASSERT_TRUE(ac.SubmitStep(0, /*id=*/1, /*px=*/105, publisher));

  Event ev{};
  ASSERT_TRUE(ring.TryPop(ev));
  EXPECT_EQ(ev.type, oep::external::TelemetryEventType::kAck);
  EXPECT_EQ(ev.order_id, 1u);
  EXPECT_EQ(ev.side, Side::kSell);
  EXPECT_EQ(ev.price, 105);
  EXPECT_EQ(ev.qty, ac.ChildQuantity(0));
  EXPECT_EQ(ev.sequence, 0u);
  EXPECT_EQ(ev.timestamp_ns, 42u);
}

}  // namespace
