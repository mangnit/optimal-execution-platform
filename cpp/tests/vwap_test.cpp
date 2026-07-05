// Unit tests for the VWAP baseline (task P5).
//
// What we prove:
//   1. Uniform volume profile ≡ TWAP (equal-sized children).
//   2. Boundary anchors: remaining_[0] == Q, remaining_[N] == 0, Σ child == Q.
//   3. Non-uniform profile splits Q by cumulative-rounded weights.
//   4. A profile that does not quite sum to 1.0 still closes out at Q.
//   5. Hot-path lookup does not touch the global allocator.
//   6. SubmitStep pushes an Ack event through TelemetryPublisher into the
//      SPSC ring.

#include "core/order_book.hpp"
#include "core/spsc_ring.hpp"
#include "exec/telemetry_publisher.hpp"
#include "exec/vwap.hpp"
#include "external/kdb_logger.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

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
using oep::exec::VwapStrategy;

TEST(Vwap, UniformProfileIsTwap) {
  std::vector<double> profile(10, 0.1);
  VwapStrategy vw({/*parent_qty=*/100,
                   /*parent_side=*/Side::kSell,
                   /*volume_profile=*/profile.data(),
                   /*num_steps=*/profile.size()});
  for (std::size_t j = 0; j < vw.num_steps(); ++j) {
    EXPECT_EQ(vw.ChildQuantity(j), 10u) << "at j=" << j;
  }
  EXPECT_EQ(vw.RemainingAfterStep(0), 100u);
  EXPECT_EQ(vw.RemainingAfterStep(vw.num_steps()), 0u);
}

TEST(Vwap, ChildSumEqualsParent) {
  std::vector<double> profile = {0.05, 0.10, 0.15, 0.20, 0.25, 0.15, 0.10};
  VwapStrategy vw({/*parent_qty=*/1'234'567,
                   /*parent_side=*/Side::kSell,
                   /*volume_profile=*/profile.data(),
                   /*num_steps=*/profile.size()});
  Quantity sum = 0;
  for (std::size_t j = 0; j < vw.num_steps(); ++j) sum += vw.ChildQuantity(j);
  EXPECT_EQ(sum, vw.parent_qty());
  EXPECT_EQ(vw.RemainingAfterStep(vw.num_steps()), 0u);
}

TEST(Vwap, CumulativeRoundingMatchesHandComputed) {
  // Q=100, profile=[0.2, 0.3, 0.4, 0.1]. Cumulative Q·Σ = 20, 50, 90, 100.
  // Rounded to integer children: 20, 30, 40, 10.
  std::vector<double> profile = {0.2, 0.3, 0.4, 0.1};
  VwapStrategy vw({100, Side::kSell, profile.data(), profile.size()});
  EXPECT_EQ(vw.ChildQuantity(0), 20u);
  EXPECT_EQ(vw.ChildQuantity(1), 30u);
  EXPECT_EQ(vw.ChildQuantity(2), 40u);
  EXPECT_EQ(vw.ChildQuantity(3), 10u);
  EXPECT_EQ(vw.RemainingAfterStep(2), 50u);
}

TEST(Vwap, ProfileNotExactlyOneStillClosesOut) {
  // Deliberately sub-unit profile — the constructor must still anchor the
  // final cumulative target at Q so Σ child == Q.
  std::vector<double> profile = {0.10, 0.20, 0.30};  // sum = 0.60
  VwapStrategy vw({/*parent_qty=*/1000,
                   /*parent_side=*/Side::kSell,
                   /*volume_profile=*/profile.data(),
                   /*num_steps=*/profile.size()});
  Quantity sum = 0;
  for (std::size_t j = 0; j < vw.num_steps(); ++j) sum += vw.ChildQuantity(j);
  EXPECT_EQ(sum, 1000u);
  EXPECT_EQ(vw.RemainingAfterStep(vw.num_steps()), 0u);
}

TEST(Vwap, HotPathHasNoHeapTraffic) {
  std::vector<double> profile(64, 1.0 / 64.0);
  VwapStrategy vw({/*parent_qty=*/1'000'000,
                   /*parent_side=*/Side::kSell,
                   /*volume_profile=*/profile.data(),
                   /*num_steps=*/profile.size()});
  volatile Quantity sink = 0;
  for (std::size_t j = 0; j < vw.num_steps(); ++j) {
    sink += vw.ChildQuantity(j);
    sink += vw.RemainingAfterStep(j);
  }

  const std::size_t n0 = TotalNews();
  const std::size_t d0 = TotalDels();
  for (int rep = 0; rep < 1000; ++rep) {
    for (std::size_t j = 0; j < vw.num_steps(); ++j) {
      sink += vw.ChildQuantity(j);
      sink += vw.RemainingAfterStep(j);
    }
  }
  EXPECT_EQ(TotalNews(), n0);
  EXPECT_EQ(TotalDels(), d0);
  (void)sink;
}

TEST(Vwap, SubmitStepEmitsAckThroughPublisher) {
  using Event = oep::external::TelemetryEvent;
  using Ring = oep::core::SpscRing<Event, 64>;
  oep::core::OrderBook book(64);
  book.PrewarmLevel(Side::kSell, 101);
  Ring ring;
  auto clock = []() noexcept -> std::uint64_t { return 99; };
  oep::exec::TelemetryPublisher publisher(book, ring, clock);

  std::vector<double> profile = {0.5, 0.5};
  VwapStrategy vw({/*parent_qty=*/40,
                   /*parent_side=*/Side::kSell,
                   /*volume_profile=*/profile.data(),
                   /*num_steps=*/profile.size()});
  ASSERT_TRUE(vw.SubmitStep(0, /*id=*/7, /*px=*/101, publisher));

  Event ev{};
  ASSERT_TRUE(ring.TryPop(ev));
  EXPECT_EQ(ev.type, oep::external::TelemetryEventType::kAck);
  EXPECT_EQ(ev.order_id, 7u);
  EXPECT_EQ(ev.side, Side::kSell);
  EXPECT_EQ(ev.price, 101);
  EXPECT_EQ(ev.qty, vw.ChildQuantity(0));
  EXPECT_EQ(ev.sequence, 0u);
  EXPECT_EQ(ev.timestamp_ns, 99u);
}

}  // namespace
