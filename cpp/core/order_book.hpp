// Intrusive LevelFIFO limit order book for the internal-clock matching engine
// (docs/architecture.md §2, task P1_order_book).
//
// Design in one paragraph. Orders are POD nodes carved out of a
// SlabAllocator<Order> — no `new`/`delete` on the hot path. At each price we
// keep a doubly-linked FIFO (LevelFIFO) so a resting order can be unlinked in
// O(1) regardless of its position in the queue. Cancel-by-id is O(1) too: an
// open-addressing hash table (id_index_) maps OrderId → Order*, uses splitmix
// for its hash, and back-shifts on erase so it never accumulates tombstones.
// The price ladder is std::map<Price, LevelFIFO> (greater<> for bids,
// less<> for asks) with a PrewarmLevel() hook that pre-inserts empty levels
// during setup so hot-path resting inserts never trigger a map node
// allocation. Empty levels are retained and skipped when scanning; that keeps
// the ladder allocation-free after warmup at the price of a bounded skip when
// matching walks through prewarmed-but-idle levels — fine for the tests and
// for real books where the density of idle levels near the touch is small.
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
#include <functional>
#include <map>
#include <utility>
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
  // `order_capacity` bounds the number of simultaneously resting orders. The
  // id-index is sized to the next power of two ≥ 2×capacity so linear probing
  // stays under a 50% load factor (fast, few probes).
  explicit OrderBook(std::size_t order_capacity)
      : slab_(order_capacity),
        id_capacity_(RoundUpPow2(
            std::max<std::size_t>(order_capacity, 1) * 2)),
        id_mask_(id_capacity_ - 1),
        id_index_(id_capacity_) {}

  OrderBook(const OrderBook&) = delete;
  OrderBook& operator=(const OrderBook&) = delete;
  OrderBook(OrderBook&&) = delete;
  OrderBook& operator=(OrderBook&&) = delete;

  // Reserve a slot in the price ladder for `price` on `side`. Meant for
  // setup: pre-inserting the levels the hot path will use prevents std::map
  // node allocations from happening once the workload starts.
  void PrewarmLevel(Side side, Price price) {
    if (side == Side::kBuy) {
      bids_.try_emplace(price);
    } else {
      asks_.try_emplace(price);
    }
  }

  // Add a limit order. Matches greedily against the opposite side while the
  // taker's price crosses, then rests any remainder as a resting limit at
  // `price`. Returns true on success; false only when the slab is exhausted
  // and a residual would otherwise need to rest (the drop-on-full policy
  // mirrors SpscRing::TryPush — never block).
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
    LevelFIFO& lvl = LevelForSide(o->side, o->price);
    UnlinkFromLevel(lvl, o);
    slab_.Deallocate(o);
    return true;
  }

  bool Cancel(OrderId id) {
    return Cancel(id, [](OrderId, Side, Price, Quantity) noexcept {});
  }

  // Observers. All O(active-empty-levels) in the worst case; for a healthy
  // book with few idle prewarmed levels near the touch, effectively O(1).
  bool has_bid() const noexcept {
    for (const auto& kv : bids_) {
      if (!kv.second.empty()) return true;
    }
    return false;
  }
  bool has_ask() const noexcept {
    for (const auto& kv : asks_) {
      if (!kv.second.empty()) return true;
    }
    return false;
  }
  Price best_bid() const noexcept {
    for (const auto& kv : bids_) {
      if (!kv.second.empty()) return kv.first;
    }
    return 0;
  }
  Price best_ask() const noexcept {
    for (const auto& kv : asks_) {
      if (!kv.second.empty()) return kv.first;
    }
    return 0;
  }
  Quantity level_qty(Side side, Price price) const noexcept {
    if (side == Side::kBuy) {
      auto it = bids_.find(price);
      return (it == bids_.end()) ? 0 : it->second.total_qty;
    }
    auto it = asks_.find(price);
    return (it == asks_.end()) ? 0 : it->second.total_qty;
  }
  std::uint32_t level_order_count(Side side, Price price) const noexcept {
    if (side == Side::kBuy) {
      auto it = bids_.find(price);
      return (it == bids_.end()) ? 0 : it->second.order_count;
    }
    auto it = asks_.find(price);
    return (it == asks_.end()) ? 0 : it->second.order_count;
  }
  const LevelFIFO* level(Side side, Price price) const noexcept {
    if (side == Side::kBuy) {
      auto it = bids_.find(price);
      return (it == bids_.end()) ? nullptr : &it->second;
    }
    auto it = asks_.find(price);
    return (it == asks_.end()) ? nullptr : &it->second;
  }
  std::size_t open_order_count() const noexcept { return slab_.in_use(); }
  std::size_t slab_capacity() const noexcept { return slab_.capacity(); }

 private:
  using BidLadder = std::map<Price, LevelFIFO, std::greater<Price>>;
  using AskLadder = std::map<Price, LevelFIFO, std::less<Price>>;

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

  LevelFIFO& LevelForSide(Side side, Price price) noexcept {
    if (side == Side::kBuy) {
      auto it = bids_.find(price);
      assert(it != bids_.end());
      return it->second;
    }
    auto it = asks_.find(price);
    assert(it != asks_.end());
    return it->second;
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
    while (remaining > 0) {
      auto it = asks_.begin();
      while (it != asks_.end() && it->second.empty()) ++it;
      if (it == asks_.end()) break;
      if (it->first > taker_price) break;  // best ask above taker's bid: done
      remaining = ConsumeLevel(taker_id, Side::kBuy, it->first, it->second,
                               remaining, on_fill);
    }
    return remaining;
  }

  template <typename FillHandler>
  Quantity MatchSell(OrderId taker_id, Price taker_price, Quantity remaining,
                     FillHandler& on_fill) {
    while (remaining > 0) {
      auto it = bids_.begin();
      while (it != bids_.end() && it->second.empty()) ++it;
      if (it == bids_.end()) break;
      if (it->first < taker_price) break;  // best bid below taker's ask: done
      remaining = ConsumeLevel(taker_id, Side::kSell, it->first, it->second,
                               remaining, on_fill);
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
    Order* o = slab_.Allocate();
    if (o == nullptr) return false;  // slab exhausted — drop the residual
    o->id = id;
    o->price = price;
    o->qty = qty;
    o->side = side;
    o->prev = nullptr;
    o->next = nullptr;
    if (side == Side::kBuy) {
      auto [it, _] = bids_.try_emplace(price);
      PushBackToLevel(it->second, o);
    } else {
      auto [it, _] = asks_.try_emplace(price);
      PushBackToLevel(it->second, o);
    }
    InsertId(id, o);
    return true;
  }

  SlabAllocator<Order> slab_;
  BidLadder bids_;
  AskLadder asks_;
  std::size_t id_capacity_;
  std::size_t id_mask_;
  std::vector<IdEntry> id_index_;
};

}  // namespace oep::core

#endif  // OEP_CORE_ORDER_BOOK_HPP_
