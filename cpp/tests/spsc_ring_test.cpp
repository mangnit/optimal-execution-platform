// Tests for oep::core::SpscRing.
//
// Focus areas (per task P1_spsc_ring):
//   1. Lock-free traits on the atomics used internally.
//   2. Cache-line padding: write_idx_, read_idx_ and dropped_messages_ live
//      on distinct 64-byte lines (false-sharing guard).
//   3. FIFO ordering across wraparound.
//   4. Drop-counter accuracy under overflow (single-threaded, deterministic).
//   5. Drops do not corrupt the FIFO of what does get through.
//   6. Concurrent producer/consumer with a slow consumer: producer never
//      blocks and drop counter matches the arithmetic identity
//         produced == received + dropped.

#include "core/spsc_ring.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

namespace {

using oep::core::kSpscCacheLine;
using oep::core::SpscRing;

// ---------------------------------------------------------------------------
// 1. Lock-free traits and basic type invariants.
// ---------------------------------------------------------------------------

TEST(SpscRingTraits, AtomicsAreAlwaysLockFree) {
  static_assert(std::atomic<std::size_t>::is_always_lock_free);
  static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
  SUCCEED();
}

TEST(SpscRingTraits, CapacityConstantsExposed) {
  using Ring = SpscRing<int, 16>;
  EXPECT_EQ(Ring::capacity(), 16u);
  EXPECT_EQ(Ring::kCapacity, 16u);
  EXPECT_EQ(Ring::kMask, 15u);
}

// ---------------------------------------------------------------------------
// 2. Cache-line padding. write_idx_, read_idx_, dropped_messages_ must each
//    be on their own line so producer/consumer never MESI-thrash a shared
//    line.
// ---------------------------------------------------------------------------

TEST(SpscRingLayout, HotAtomicsOnDistinctCacheLines) {
  using Ring = SpscRing<std::uint64_t, 16>;
  const auto w = Ring::write_index_offset_for_testing();
  const auto r = Ring::read_index_offset_for_testing();
  const auto d = Ring::dropped_offset_for_testing();

  EXPECT_EQ(w % kSpscCacheLine, 0u);
  EXPECT_EQ(r % kSpscCacheLine, 0u);
  EXPECT_EQ(d % kSpscCacheLine, 0u);

  const auto line_of = [](std::size_t off) { return off / kSpscCacheLine; };
  EXPECT_NE(line_of(w), line_of(r));
  EXPECT_NE(line_of(w), line_of(d));
  EXPECT_NE(line_of(r), line_of(d));
}

TEST(SpscRingLayout, WholeStructIsCacheAligned) {
  using Ring = SpscRing<std::uint64_t, 16>;
  static_assert(alignof(Ring) >= kSpscCacheLine);
  Ring ring;
  const auto addr = reinterpret_cast<std::uintptr_t>(&ring);
  EXPECT_EQ(addr % kSpscCacheLine, 0u);
}

// ---------------------------------------------------------------------------
// 3. FIFO ordering and wraparound.
// ---------------------------------------------------------------------------

TEST(SpscRingFifo, EmptyPopReturnsFalse) {
  SpscRing<int, 8> ring;
  int out = 42;
  EXPECT_FALSE(ring.TryPop(out));
  EXPECT_EQ(out, 42);  // unchanged
  EXPECT_EQ(ring.size_approx(), 0u);
  EXPECT_TRUE(ring.empty_approx());
}

TEST(SpscRingFifo, PushPopSingleValue) {
  SpscRing<int, 8> ring;
  EXPECT_TRUE(ring.TryPush(7));
  EXPECT_EQ(ring.size_approx(), 1u);
  int out = 0;
  EXPECT_TRUE(ring.TryPop(out));
  EXPECT_EQ(out, 7);
  EXPECT_EQ(ring.size_approx(), 0u);
}

TEST(SpscRingFifo, PreservesOrderAcrossWraparound) {
  // 10 push/pop rounds through a capacity-8 ring => guaranteed wraparound.
  SpscRing<int, 8> ring;
  int expect = 0;
  for (int round = 0; round < 32; ++round) {
    for (int i = 0; i < 5; ++i) {
      ASSERT_TRUE(ring.TryPush(expect + i));
    }
    for (int i = 0; i < 5; ++i) {
      int out = -1;
      ASSERT_TRUE(ring.TryPop(out));
      EXPECT_EQ(out, expect + i);
    }
    expect += 5;
  }
  EXPECT_EQ(ring.dropped_messages(), 0u);
}

TEST(SpscRingFifo, HoldsExactlyCapacityElements) {
  SpscRing<int, 8> ring;
  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(ring.TryPush(i));
  }
  EXPECT_EQ(ring.size_approx(), 8u);
  // Next push must be dropped, not blocked.
  EXPECT_FALSE(ring.TryPush(999));
  EXPECT_EQ(ring.dropped_messages(), 1u);
  // Drain and confirm order.
  for (int i = 0; i < 8; ++i) {
    int out = -1;
    ASSERT_TRUE(ring.TryPop(out));
    EXPECT_EQ(out, i);
  }
  int sink = 0;
  EXPECT_FALSE(ring.TryPop(sink));
}

// ---------------------------------------------------------------------------
// 4. Overflow drop-counter accuracy (deterministic, single-threaded).
// ---------------------------------------------------------------------------

TEST(SpscRingDrops, CounterExactUnderMassiveOverflow) {
  constexpr std::size_t kCap = 16;
  SpscRing<std::uint32_t, kCap> ring;
  constexpr std::size_t kAttempts = 10'000;
  std::size_t accepted = 0;
  for (std::size_t i = 0; i < kAttempts; ++i) {
    if (ring.TryPush(static_cast<std::uint32_t>(i))) {
      ++accepted;
    }
  }
  EXPECT_EQ(accepted, kCap);
  EXPECT_EQ(ring.dropped_messages(), kAttempts - kCap);
  EXPECT_EQ(ring.size_approx(), kCap);
}

TEST(SpscRingDrops, ProducerNeverStalls) {
  // Timing-based smoke: with a full ring, N failed pushes in a row must all
  // complete in bounded time. The point is to catch any accidental
  // spin/back-off code path the impl might grow later.
  SpscRing<int, 4> ring;
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(ring.TryPush(i));
  }
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < 100'000; ++i) {
    EXPECT_FALSE(ring.TryPush(i));
  }
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  EXPECT_EQ(ring.dropped_messages(), 100'000u);
}

TEST(SpscRingDrops, DropsDoNotCorruptFifoOfSurvivors) {
  // Push 8, then attempt to push 8 more (all dropped), then pop 8, then push
  // 8 fresh, pop 8. The pop sequence must be 0..7 followed by 100..107,
  // proving that the dropped writes never smeared into the tail.
  constexpr std::size_t kCap = 8;
  SpscRing<int, kCap> ring;
  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    ASSERT_TRUE(ring.TryPush(i));
  }
  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    EXPECT_FALSE(ring.TryPush(9000 + i));
  }
  EXPECT_EQ(ring.dropped_messages(), kCap);

  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    int out = -1;
    ASSERT_TRUE(ring.TryPop(out));
    EXPECT_EQ(out, i);
  }
  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    ASSERT_TRUE(ring.TryPush(100 + i));
  }
  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    int out = -1;
    ASSERT_TRUE(ring.TryPop(out));
    EXPECT_EQ(out, 100 + i);
  }
  EXPECT_EQ(ring.dropped_messages(), kCap);
}

TEST(SpscRingDrops, InterleavedPushPopKeepsCounterMonotone) {
  // Alternate near-full states: fill, pop one, push one (ok), push one (dropped)
  // repeated. Verifies the counter only advances on genuine drops.
  constexpr std::size_t kCap = 4;
  SpscRing<int, kCap> ring;
  for (int i = 0; i < static_cast<int>(kCap); ++i) {
    ASSERT_TRUE(ring.TryPush(i));
  }
  std::uint64_t expected_drops = 0;
  int next_value = static_cast<int>(kCap);
  for (int round = 0; round < 1000; ++round) {
    int out = -1;
    ASSERT_TRUE(ring.TryPop(out));
    ASSERT_TRUE(ring.TryPush(next_value++));   // fills the slot back
    EXPECT_FALSE(ring.TryPush(next_value++));  // ring full again -> dropped
    ++expected_drops;
    EXPECT_EQ(ring.dropped_messages(), expected_drops);
  }
}

// ---------------------------------------------------------------------------
// 5. Concurrent producer/consumer.
// ---------------------------------------------------------------------------

TEST(SpscRingConcurrent, SlowConsumerNeverBlocksProducer) {
  // Producer emits a dense stream with no back-off. Consumer sleeps briefly
  // between pops so the ring saturates and drops occur. Post-condition:
  // produced == received + dropped, and the received sequence is strictly
  // monotonically increasing (proves no reordering across the boundary).
  constexpr std::size_t kCap = 64;
  SpscRing<std::uint64_t, kCap> ring;
  constexpr std::uint64_t kTotal = 200'000;

  std::atomic<bool> producer_done{false};
  std::vector<std::uint64_t> received;
  received.reserve(kTotal);

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kTotal; ++i) {
      // Deliberately do not spin-wait; a full ring means the value drops.
      (void)ring.TryPush(i);
    }
    producer_done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    std::uint64_t out;
    while (true) {
      if (ring.TryPop(out)) {
        received.push_back(out);
        // Occasional stall to force the ring full.
        if ((received.size() & 0x3FF) == 0) {
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
      } else if (producer_done.load(std::memory_order_acquire) &&
                 ring.empty_approx()) {
        break;
      }
    }
  });

  producer.join();
  consumer.join();

  const std::uint64_t dropped = ring.dropped_messages();
  EXPECT_EQ(received.size() + dropped, kTotal)
      << "produced=" << kTotal << " received=" << received.size()
      << " dropped=" << dropped;
  EXPECT_GT(dropped, 0u) << "test did not exercise the drop path";

  // Strictly monotonic: what got through must have been in issue order.
  for (std::size_t i = 1; i < received.size(); ++i) {
    ASSERT_LT(received[i - 1], received[i])
        << "reordering at index " << i;
  }
}

TEST(SpscRingConcurrent, RetryingProducerLosesNothingAndOrdered) {
  // Symmetric sanity: if the producer is willing to retry on full, every
  // value gets through in issue order. Failed TryPush attempts still tick
  // the drop counter (by design — the ring cannot distinguish "will retry"
  // from "gave up"), so we only assert on the delivered stream.
  constexpr std::size_t kCap = 128;
  SpscRing<std::uint64_t, kCap> ring;
  constexpr std::uint64_t kTotal = 500'000;

  std::atomic<bool> producer_done{false};
  std::vector<std::uint64_t> received;
  received.reserve(kTotal);

  std::thread producer([&] {
    for (std::uint64_t i = 0; i < kTotal; ++i) {
      while (!ring.TryPush(i)) {
        std::this_thread::yield();
      }
    }
    producer_done.store(true, std::memory_order_release);
  });

  std::thread consumer([&] {
    std::uint64_t out;
    while (true) {
      if (ring.TryPop(out)) {
        received.push_back(out);
      } else if (producer_done.load(std::memory_order_acquire) &&
                 ring.empty_approx()) {
        break;
      }
    }
  });

  producer.join();
  consumer.join();

  ASSERT_EQ(received.size(), kTotal);
  for (std::uint64_t i = 0; i < kTotal; ++i) {
    ASSERT_EQ(received[i], i) << "reorder or loss at " << i;
  }
}

}  // namespace
