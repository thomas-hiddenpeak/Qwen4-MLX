#include "immutable_page_pool.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>

using namespace anern::paged;

namespace {
std::size_t checks = 0;
void require(bool value, const char* message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}
template <class Exception, class F> void rejects(F&& f, const char* message) {
  bool rejected = false;
  try { f(); } catch (const Exception&) { rejected = true; }
  require(rejected, message);
}

// CPU payload witness: apply exactly the planned page copies and input writes.
// This exercises metadata routing, not any MLX/Metal behavior.
using Arena = std::vector<std::vector<std::int64_t>>;
PageList append(PagePool& pool, Arena& arena, const PageList& source,
                std::size_t rows, std::int64_t start) {
  auto tx = pool.planAppend(source, rows);
  for (const auto& copy : tx.copies())
    std::copy_n(arena.at(copy.source.slot).begin(), copy.rows,
                arena.at(copy.destination.slot).begin());
  std::size_t written = 0;
  for (const auto& write : tx.writes()) {
    require(write.rows > 0 && write.destinationRow + write.rows <= kPageRows,
            "write segments remain within one physical page");
    require(write.sourceRow == written, "write segments cover contiguous input rows");
    for (std::size_t i = 0; i < write.rows; ++i)
      arena.at(write.destination.slot).at(write.destinationRow + i) =
          start + static_cast<std::int64_t>(write.sourceRow + i);
    written += write.rows;
  }
  require(written == rows, "all appended rows are written once");
  return tx.commit();
}
std::vector<std::int64_t> rows(const Arena& arena, const PageList& list) {
  std::vector<std::int64_t> result;
  for (std::size_t i = 0; i < list.tokenCount(); ++i)
    result.push_back(arena.at(list.pageIDs().at(i / kPageRows)).at(i % kPageRows));
  return result;
}
std::vector<std::int64_t> sequence(std::size_t count, std::int64_t start) {
  std::vector<std::int64_t> result(count);
  std::iota(result.begin(), result.end(), start);
  return result;
}

void forkAndPayload() {
  PagePool pool(8);
  Arena arena(8, std::vector<std::int64_t>(kPageRows, -1));
  auto base = append(pool, arena, pool.empty(), 65, 0);
  auto sibling = base;
  require(pool.stats().uniqueLivePages == 3, "fork allocates no physical pages");
  auto left = append(pool, arena, base, 31, 65);
  auto right = append(pool, arena, sibling, 33, 1000);
  require(left.pageIDs()[0] == base.pageIDs()[0] && left.pageIDs()[1] == base.pageIDs()[1],
          "full pages are shared without copying");
  require(left.pageIDs()[2] != base.pageIDs()[2] && right.pageIDs()[2] != base.pageIDs()[2],
          "both fork tails receive new physical versions");
  require(rows(arena, base) == sequence(65, 0), "parent payload stays immutable");
  require(rows(arena, left) == sequence(96, 0), "left fork payload is correct");
  auto expectedRight = sequence(65, 0);
  auto suffix = sequence(33, 1000);
  expectedRight.insert(expectedRight.end(), suffix.begin(), suffix.end());
  require(rows(arena, right) == expectedRight, "right fork payload is independent");
  auto stats = pool.stats();
  require(stats.uniqueLivePages == 6 && stats.publishedPages == 6 && stats.reservedPages == 0,
          "unique pages count shared full pages once");
  require(stats.committedPlannedTailCopies == 2 && stats.committedPlannedCopiedRows == 2 &&
          stats.committedPlannedCopiedBytes == 2 * kRowPairBytes,
          "copy counters include only old tail rows");
  require(base.logicalBytes() == 65 * kRowPairBytes &&
          stats.uniquePhysicalBytes == 6 * kPagePairBytes,
          "logical payload and unique physical page accounting stay distinct");
  base = {}; sibling = {};
  require(pool.stats().uniqueLivePages == 5, "old tail returns after last parent holder releases");
  left = {};
  require(pool.stats().uniqueLivePages == 4, "shared full pages survive sibling release");
  pool.validate(right);
  right = {};
  require(pool.stats().freePages == 8, "all slots reclaim after final fork releases");
}

void reservationAndRollback() {
  PagePool pool(3);
  Arena arena(3, std::vector<std::int64_t>(kPageRows, -1));
  auto base = append(pool, arena, pool.empty(), 64, 0);
  const auto before = pool.stats();
  rejects<PagePoolExhausted>([&] { (void)pool.planAppend(base, 33); },
                            "multi-page OOM rejects the whole reservation");
  auto after = pool.stats();
  require(after.freePages == before.freePages && after.uniqueLivePages == before.uniqueLivePages &&
          after.committedPlans == before.committedPlans,
          "OOM leaves all allocation and commit counters unchanged");
  {
    auto tx = pool.planAppend(base, 1);
    require(tx.copies().empty(), "aligned append does not copy any old page");
    require(pool.stats().reservedPages == 1, "unpublished append slot is reserved");
    tx.rollback(); tx.rollback();
    require(pool.stats().reservedPages == 0, "rollback is idempotent and frees unused reservation");
    rejects<std::logic_error>([&] { (void)tx.commit(); }, "commit after rollback is rejected");
  }
  {
    auto tx = pool.planAppend(base, 1);
    auto moved = std::move(tx);
    rejects<std::logic_error>([&] { (void)tx.commit(); }, "moved-from transaction cannot commit");
    auto next = moved.commit();
    const auto committed = pool.stats().committedPlans;
    rejects<std::logic_error>([&] { (void)moved.commit(); }, "double commit is rejected");
    require(pool.stats().committedPlans == committed, "double commit cannot inflate counters");
    pool.validate(next);
  }
  require(pool.stats().uniqueLivePages == 2, "transaction destruction reclaims unretained output");
}

void completionPinsAndReuse() {
  PagePool pool(3);
  Arena arena(3, std::vector<std::int64_t>(kPageRows, -1));
  auto source = append(pool, arena, pool.empty(), 31, 0);
  const auto oldAddress = source.pageAddresses()[0];
  auto tx = pool.planAppend(source, 1);
  auto pin = tx.lifetimePin();
  auto destination = tx.commit();
  source = {}; destination = {}; tx.rollback();
  require(pool.stats().uniqueLivePages == 2, "completion pin retains both read source and write destination");
  pin.reset();
  require(pool.stats().uniqueLivePages == 0, "completion releases physical page leases");
  auto reused = append(pool, arena, pool.empty(), 96, 100);
  const auto found = std::find_if(reused.pageAddresses().begin(), reused.pageAddresses().end(),
      [&](const auto& address) { return address.slot == oldAddress.slot; });
  require(found != reused.pageAddresses().end() && found->generation > oldAddress.generation,
          "reclaimed physical slot gets a new generation");
  reused = {};
  source = append(pool, arena, pool.empty(), 31, 0);
  auto cancelled = pool.planAppend(source, 1);
  auto cancelledPin = cancelled.lifetimePin();
  cancelled.rollback(); source = {};
  require(pool.stats().reservedPages == 1 && pool.stats().publishedPages == 1,
          "rollback cannot reuse slots still pinned by in-flight work");
  cancelledPin.reset();
  require(pool.stats().uniqueLivePages == 0, "cancel completion releases source and unpublished destination");
}

void boundariesAndIdentity() {
  for (const auto initial : {31U, 32U, 33U, 2051U, 2052U, 11057U}) {
    const auto capacity = (initial + 100U) / kPageRows + 8;
    PagePool pool(capacity);
    Arena arena(capacity, std::vector<std::int64_t>(kPageRows, -1));
    auto state = append(pool, arena, pool.empty(), initial, 0);
    for (std::size_t i = 0; i < 67; ++i) {
      state = append(pool, arena, state, 1, initial + i);
      pool.validate(state);
    }
    require(rows(arena, state) == sequence(initial + 67, 0),
            "bulk import and tail rollover preserve logical row sequence");
    require(pool.stats().uniqueLivePages == state.pageCount(),
            "old versions reclaim after each unshared state replacement");
    require(pool.stats().committedPlannedCopiedRows <= 67 * 31,
            "append never plans a whole-prefix copy");
  }
  PagePool first(2), other(2);
  rejects<std::invalid_argument>([&] { (void)other.planAppend(first.empty(), 1); },
                                 "cross-pool state cannot supply raw page IDs");
  rejects<std::invalid_argument>([&] { (void)first.planAppend(PageList{}, 1); },
                                 "unbound list is rejected");
  rejects<std::invalid_argument>([&] { (void)first.planAppend(first.empty(), 0); },
                                 "empty append is rejected");
  rejects<PagePoolExhausted>([&] { (void)first.planAppend(first.empty(), 65); },
                            "logical capacity is bounded before allocation");
  std::shared_ptr<const void> pin;
  {
    PagePool temporary(1);
    auto tx = temporary.planAppend(temporary.empty(), 1);
    auto state = tx.commit();
    pin = state.lifetimePin();
  }
  pin.reset(); // Reclaim remains valid after the public PagePool object dies.
  require(true, "leases keep their pool alive until final release");
}

void concurrentLeaseRelease() {
  PagePool pool(8);
  auto initial = pool.planAppend(pool.empty(), 31);
  auto source = initial.commit();
  initial.rollback();
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(4);
  for (std::size_t i = 0; i < errors.size(); ++i) {
    threads.emplace_back([&, i] {
      try {
        for (int iteration = 0; iteration < 64; ++iteration) {
          auto tx = pool.planAppend(source, 1);
          auto pin = tx.lifetimePin();
          auto output = tx.commit();
          pool.validate(output);
          tx.rollback();
          output = {};
          pin.reset();
        }
      } catch (...) { errors[i] = std::current_exception(); }
    });
  }
  for (auto& thread : threads) thread.join();
  for (const auto& error : errors) if (error) std::rethrow_exception(error);
  require(pool.stats().uniqueLivePages == 1 && pool.stats().committedPlans == 257,
          "concurrent completion releases preserve shared source and exact commit count");
  source = {};
  require(pool.stats().freePages == 8, "concurrent lease reuse leaves no leaked slots");
}
} // namespace

int main() {
  try {
    forkAndPayload(); reservationAndRollback(); completionPinsAndReuse(); boundariesAndIdentity();
    concurrentLeaseRelease();
    std::cout << "{\"status\":\"pass\",\"checks\":" << checks
              << ",\"page_rows\":32,\"page_pair_bytes\":65536}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "metadata test failed after " << checks << " checks: " << error.what() << '\n';
    return 1;
  }
}
