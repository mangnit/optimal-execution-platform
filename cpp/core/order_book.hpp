// Intrusive LevelFIFO limit order book for the internal-clock matching engine
// (docs/architecture.md §2, task P1_order_book).
//
// Design in one paragraph. Orders are POD nodes carved out of a
// SlabAllocator<Order> — no `new`/`delete` on the hot path. At each price we
// keep a doubly-linked FIFO (LevelFIFO) so a resting order can be unlinked in
// O(1) regardless of its position in the queue. Cancel-by-id is O(1) too: an
// open-addressing hash table (id_index_) maps OrderId → Order*, uses splitmix
// for its hash, and back-shifts on erase so it never accumulates tombstones.
// The price ladder is a direct-indexed array: one contiguous
// std::vector<LevelFIFO> per side spanning a fixed price band
// [min_price, max_price], preallocated at construction. Level lookup is a
// single subtract-and-index (no tree walk, no hashing, no allocation, cache
// lines laid out in price order), and the best bid/ask are tracked
// incrementally: maintained on insert, and repaired after a level empties by
// a short contiguous scan toward worse prices — bounded by the band and, in
// practice, by the distance to the next populated level near the touch. A
// per-side live-level counter short-circuits the scan entirely on an empty
// side, making has_bid()/best_bid() O(1) reads. Prices outside the band
// cannot rest (the residual is dropped and AddLimit returns false — the same
// drop-on-full policy as slab exhaustion); matching against resting orders
// is unaffected since it only compares prices. This replaces the original
// std::map ladder, whose node-hopping best-of scans dominated the
// AddLimit p99 (docs/latency_report.md).
// Matching is standard price/time priority; fills are delivered synchronously
// via a templated FillHandler so the caller can plumb them straight into the
// SPSC ring without virtual dispatch.

#ifndef OEP_CORE_ORDER_BOOK_HPP_
#define OEP_CORE_ORDER_BOOK_HPP_

#include "core/slab_allocator.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace oep::core {

using OrderId = std::uint64_t;
using Price = std::int64_t;       // tick-integer price; no floats on the book
using Quantity = std::uint64_t;

enum class Side : std::uint8_t { kBuy = 0, kSell = 1 };

struct Fill {
  OrderId maker_id;
  OrderId taker_id;
  Price price;         // maker's resting price — standard convention
  Quantity qty;
  Side taker_side;     // side of the aggressor; maker is the opposite
};

// Intrusive order node. `prev`/`next` thread the LevelFIFO doubly-linked list;
// no separate list-node type, so the queue owns no allocations of its own.
struct Order {
  OrderId id;
  Price price;
  Quantity qty;        // remaining, decremented as maker fills accrue
  Side side;
  Order* prev;
  Order* next;
};

// One FIFO queue at a single price. Head is the earliest-arrived resting
// order; new arrivals push at tail. total_qty tracks the sum of remaining
// quantities so best-of / depth queries are O(1).
struct LevelFIFO {
  Order* head = nullptr;
  Order* tail = nullptr;
  Quantity total_qty = 0;
  std::uint32_t order_count = 0;

  bool empty() const noexcept { return head == nullptr; }
};

class OrderBook {
 public:
  // Default price band for the direct-indexed ladder. Wide enough for every
  // in-tree workload (sim/env prices ~80–120, bench ladders at ~10 000) at
  // a construction-time cost of band × 2 sides × sizeof(LevelFIFO) ≈ 1 MiB.
  // Narrow it via the constructor for tighter cache residency.
  static constexpr Price kDefaultMinPrice = 0;
  static constexpr Price kDefaultMaxPrice = 16'383;

  // `order_capacity` bounds the number of simultaneously resting orders. The
  // id-index is sized to the next power of two ≥ 2×capacity so linear probing
  // stays under a 50% load factor (fast, few probes). `[min_price, max_price]`
  // is the inclusive band the ladder can rest orders in; both vectors are
  // fully allocated here so the hot path never allocates.
  explicit OrderBook(std::size_t order_capacity,
                     Price min_price = kDefaultMinPrice,
                     Price max_price = kDefaultMaxPrice)
      : slab_(order_capacity),
        min_price_(min_price),
        max_price_(max_price),
        bid_levels_(static_cast<std::size_t>(max_price - min_price + 1)),
        ask_levels_(static_cast<std::size_t>(max_price - min_price + 1)),
        id_capacity_(RoundUpPow2(
            std::max<std::size_t>(order_capacity, 1) * 2)),
        id_mask_(id_capacity_ - 1),
        id_index_(id_capacity_) {
    assert(max_price >= min_price);
  }

  OrderBook(const OrderBook&) = delete;
  OrderBook& operator=(const OrderBook&) = delete;
  OrderBook(OrderBook&&) = delete;
  OrderBook& operator=(OrderBook&&) = delete;

  // Setup-time hook, retained for API compatibility with the std::map ladder
  // (which needed levels pre-inserted to avoid hot-path node allocations).
  // The array ladder preallocates every in-band level at construction, so
  // this is now a no-op beyond validating the price is inside the band.
  void PrewarmLevel(Side side, Price price) {
    (void)side;
    (void)price;
    assert(InBand(price));
  }

  // Add a limit order. Matches greedily against the opposite side while the
  // taker's price crosses, then rests any remainder as a resting limit at
  // `price`. Returns true on success; false only when a residual would need
  // to rest and cannot — slab exhausted, or `price` outside the ladder band
  // (the drop-on-full policy mirrors SpscRing::TryPush — never block).
  //
  // FillHandler is called synchronously as `on_fill(const Fill&)` for each
  // maker/taker pair. Templated so a lambda or SPSC-push functor is inlined
  // without virtual dispatch.
  template <typename FillHandler>
  bool AddLimit(OrderId id, Side side, Price price, Quantity qty,
                FillHandler&& on_fill) {
    assert(qty > 0);
    Quantity remaining = qty;
    if (side == Side::kBuy) {
      remaining = MatchBuy(id, price, remaining, on_fill);
    } else {
      remaining = MatchSell(id, price, remaining, on_fill);
    }
    if (remaining == 0) {
      return true;
    }
    return Rest(id, side, price, remaining);
  }

  // O(1) cancel: hash-table lookup + intrusive unlink + slab deallocate.
  // Returns false if the id is unknown (e.g. already fully filled or never
  // rested).
  //
  // The templated overload notifies the caller of the resting state we are
  // about to destroy *before* deallocation, so a telemetry publisher can
  // emit a Cancel event carrying the correct (side, price, residual qty).
  // `on_cancel` is invoked synchronously as
  //   on_cancel(OrderId, Side, Price, Quantity)
  // and templated to inline without virtual dispatch — same pattern as the
  // FillHandler on AddLimit.
  template <typename OnCancel>
  bool Cancel(OrderId id, OnCancel&& on_cancel) {
    Order* o = LookupAndEraseId(id);
    if (o == nullptr) {
      return false;
    }
    on_cancel(o->id, o->side, o->price, o->qty);
    LevelFIFO& lvl = LevelAt(o->side, o->price);
    UnlinkFromLevel(lvl, o);
    if (lvl.empty()) {
      MarkLevelEmptied(o->side, o->price);
    }
    slab_.Deallocate(o);
    return true;
  }

  bool Cancel(OrderId id) {
    return Cancel(id, [](OrderId, Side, Price, Quantity) noexcept {});
  }

  // Observers. All O(1): the best prices are maintained incrementally by
  // insert / cancel / match, never recomputed by scanning on read.
  bool has_bid() const noexcept { return live_bid_levels_ > 0; }
  bool has_ask() const noexcept { return live_ask_levels_ > 0; }
  Price best_bid() const noexcept {
    return live_bid_levels_ > 0 ? best_bid_px_ : 0;
  }
  Price best_ask() const noexcept {
    return live_ask_levels_ > 0 ? best_ask_px_ : 0;
  }
  Quantity level_qty(Side side, Price price) const noexcept {
    if (!InBand(price)) return 0;
    return (side == Side::kBuy ? bid_levels_ : ask_levels_)[Idx(price)]
        .total_qty;
  }
  std::uint32_t level_order_count(Side side, Price price) const noexcept {
    if (!InBand(price)) return 0;
    return (side == Side::kBuy ? bid_levels_ : ask_levels_)[Idx(price)]
        .order_count;
  }
  const LevelFIFO* level(Side side, Price price) const noexcept {
    if (!InBand(price)) return nullptr;
    return &(side == Side::kBuy ? bid_levels_ : ask_levels_)[Idx(price)];
  }
  std::size_t open_order_count() const noexcept { return slab_.in_use(); }
  std::size_t slab_capacity() const noexcept { return slab_.capacity(); }
  Price min_price() const noexcept { return min_price_; }
  Price max_price() const noexcept { return max_price_; }

 private:
  struct IdEntry {
    OrderId id = 0;
    Order* ptr = nullptr;
  };

  static constexpr std::size_t RoundUpPow2(std::size_t n) noexcept {
    std::size_t p = 1;
    while (p < n) p <<= 1;
    return p;
  }

  // splitmix64 finaliser. Cheap, high-quality avalanche for dense integer
  // ids — the input pattern id_index_ actually sees in this project.
  static constexpr std::uint64_t MixId(OrderId id) noexcept {
    std::uint64_t x = id + 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
  }

  bool InBand(Price px) const noexcept {
    return px >= min_price_ && px <= max_price_;
  }
  std::size_t Idx(Price px) const noexcept {
    return static_cast<std::size_t>(px - min_price_);
  }
  LevelFIFO& LevelAt(Side side, Price px) noexcept {
    assert(InBand(px));
    return (side == Side::kBuy ? bid_levels_ : ask_levels_)[Idx(px)];
  }

  // A level transitioned empty → populated. Maintain the incremental best:
  // a bid improves the touch when higher, an ask when lower.
  void MarkLevelLive(Side side, Price px) noexcept {
    if (side == Side::kBuy) {
      if (live_bid_levels_ == 0 || px > best_bid_px_) {
        best_bid_px_ = px;
      }
      ++live_bid_levels_;
    } else {
      if (live_ask_levels_ == 0 || px < best_ask_px_) {
        best_ask_px_ = px;
      }
      ++live_ask_levels_;
    }
  }

  // A level transitioned populated → empty. If it carried the touch, repair
  // by scanning toward worse prices over the contiguous ladder. The live
  // counter guarantees termination before the band edge whenever any level
  // remains; an emptied side skips the scan entirely.
  void MarkLevelEmptied(Side side, Price px) noexcept {
    if (side == Side::kBuy) {
      --live_bid_levels_;
      if (live_bid_levels_ > 0 && px == best_bid_px_) {
        Price p = px;
        do {
          --p;
        } while (bid_levels_[Idx(p)].empty());
        best_bid_px_ = p;
      }
    } else {
      --live_ask_levels_;
      if (live_ask_levels_ > 0 && px == best_ask_px_) {
        Price p = px;
        do {
          ++p;
        } while (ask_levels_[Idx(p)].empty());
        best_ask_px_ = p;
      }
    }
  }

  void InsertId(OrderId id, Order* o) noexcept {
    // Load factor is bounded to 50% by construction, so an empty slot always
    // exists — the loop terminates.
    std::size_t slot = MixId(id) & id_mask_;
    while (id_index_[slot].ptr != nullptr) {
      slot = (slot + 1) & id_mask_;
    }
    id_index_[slot] = IdEntry{id, o};
  }

  // Look up an id and erase its slot. Back-shift deletion (Knuth 6.4-R):
  // scan forward through the cluster; an entry is movable into the hole iff
  // its probe distance from home is >= its distance from the hole, i.e. its
  // home lies cyclically at-or-before the hole. Unmovable entries (home
  // strictly after the hole) are skipped and the scan continues to the end
  // of the cluster. Keeps the table tombstone-free so lookup cost stays
  // bounded.
  //
  // The movability test is load-bearing. An earlier version broke out of
  // the shift at the first entry sitting on its home slot, which strands
  // any displaced entries further along the cluster behind the new hole —
  // orphaned (unreachable by lookup, so never erased). Under sustained
  // insert/erase churn the orphans accumulate until the table saturates,
  // and InsertId (which requires an empty slot to terminate) then spins
  // forever. Found by the fused hot-path benchmark; regression-guarded by
  // OrderBookTest.IdIndexSurvivesSustainedChurn.
  Order* LookupAndEraseId(OrderId id) noexcept {
    const std::size_t start = MixId(id) & id_mask_;
    std::size_t slot = start;
    for (std::size_t probes = 0; probes < id_capacity_; ++probes) {
      IdEntry& e = id_index_[slot];
      if (e.ptr == nullptr) {
        return nullptr;  // empty slot => not present
      }
      if (e.id == id) {
        Order* o = e.ptr;
        std::size_t hole = slot;
        std::size_t next = (hole + 1) & id_mask_;
        while (id_index_[next].ptr != nullptr) {
          const std::size_t home = MixId(id_index_[next].id) & id_mask_;
          if (((next - home) & id_mask_) >= ((next - hole) & id_mask_)) {
            id_index_[hole] = id_index_[next];
            hole = next;
          }
          next = (next + 1) & id_mask_;
        }
        id_index_[hole] = IdEntry{};
        return o;
      }
      slot = (slot + 1) & id_mask_;
    }
    return nullptr;
  }

  static void UnlinkFromLevel(LevelFIFO& lvl, Order* o) noexcept {
    if (o->prev != nullptr) {
      o->prev->next = o->next;
    } else {
      lvl.head = o->next;
    }
    if (o->next != nullptr) {
      o->next->prev = o->prev;
    } else {
      lvl.tail = o->prev;
    }
    lvl.total_qty -= o->qty;
    --lvl.order_count;
  }

  static void PushBackToLevel(LevelFIFO& lvl, Order* o) noexcept {
    o->prev = lvl.tail;
    o->next = nullptr;
    if (lvl.tail != nullptr) {
      lvl.tail->next = o;
    } else {
      lvl.head = o;
    }
    lvl.tail = o;
    lvl.total_qty += o->qty;
    ++lvl.order_count;
  }

  template <typename FillHandler>
  Quantity MatchBuy(OrderId taker_id, Price taker_price, Quantity remaining,
                    FillHandler& on_fill) {
    while (remaining > 0 && live_ask_levels_ > 0) {
      const Price px = best_ask_px_;
      if (px > taker_price) break;  // best ask above taker's bid: done
      LevelFIFO& lvl = ask_levels_[Idx(px)];
      remaining = ConsumeLevel(taker_id, Side::kBuy, px, lvl, remaining,
                               on_fill);
      if (lvl.empty()) {
        MarkLevelEmptied(Side::kSell, px);
      }
    }
    return remaining;
  }

  template <typename FillHandler>
  Quantity MatchSell(OrderId taker_id, Price taker_price, Quantity remaining,
                     FillHandler& on_fill) {
    while (remaining > 0 && live_bid_levels_ > 0) {
      const Price px = best_bid_px_;
      if (px < taker_price) break;  // best bid below taker's ask: done
      LevelFIFO& lvl = bid_levels_[Idx(px)];
      remaining = ConsumeLevel(taker_id, Side::kSell, px, lvl, remaining,
                               on_fill);
      if (lvl.empty()) {
        MarkLevelEmptied(Side::kBuy, px);
      }
    }
    return remaining;
  }

  // Drain the head of a level FIFO against the taker until either the level
  // empties or the taker is filled. Fully-consumed makers are removed from
  // both the level list and the id index and returned to the slab.
  template <typename FillHandler>
  Quantity ConsumeLevel(OrderId taker_id, Side taker_side, Price maker_price,
                        LevelFIFO& lvl, Quantity remaining,
                        FillHandler& on_fill) {
    while (remaining > 0 && lvl.head != nullptr) {
      Order* maker = lvl.head;
      const Quantity fill = std::min<Quantity>(remaining, maker->qty);
      on_fill(Fill{maker->id, taker_id, maker_price, fill, taker_side});
      maker->qty -= fill;
      lvl.total_qty -= fill;
      remaining -= fill;
      if (maker->qty == 0) {
        // Pop the head; total_qty already reflects the fill, so don't route
        // through UnlinkFromLevel (which would subtract maker->qty == 0).
        lvl.head = maker->next;
        if (lvl.head == nullptr) {
          lvl.tail = nullptr;
        } else {
          lvl.head->prev = nullptr;
        }
        --lvl.order_count;
        LookupAndEraseId(maker->id);
        slab_.Deallocate(maker);
      }
    }
    return remaining;
  }

  bool Rest(OrderId id, Side side, Price price, Quantity qty) {
    if (!InBand(price)) return false;  // outside the ladder band — drop
    Order* o = slab_.Allocate();
    if (o == nullptr) return false;  // slab exhausted — drop the residual
    o->id = id;
    o->price = price;
    o->qty = qty;
    o->side = side;
    o->prev = nullptr;
    o->next = nullptr;
    LevelFIFO& lvl = LevelAt(side, price);
    const bool was_empty = lvl.empty();
    PushBackToLevel(lvl, o);
    if (was_empty) {
      MarkLevelLive(side, price);
    }
    InsertId(id, o);
    return true;
  }

  SlabAllocator<Order> slab_;

  // Direct-indexed price ladder: level i holds price min_price_ + i.
  // Both sides fully preallocated at construction; no hot-path allocation.
  Price min_price_;
  Price max_price_;
  std::vector<LevelFIFO> bid_levels_;
  std::vector<LevelFIFO> ask_levels_;

  // Incrementally-maintained touch. `best_*_px_` is meaningful only while
  // the matching live counter is non-zero.
  Price best_bid_px_ = 0;
  Price best_ask_px_ = 0;
  std::size_t live_bid_levels_ = 0;
  std::size_t live_ask_levels_ = 0;

  std::size_t id_capacity_;
  std::size_t id_mask_;
  std::vector<IdEntry> id_index_;
};

}  // namespace oep::core

#endif  // OEP_CORE_ORDER_BOOK_HPP_
