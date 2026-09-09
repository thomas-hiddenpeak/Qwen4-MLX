#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace anern::paged {

// One layer: 2 KV heads * 32 rows * 256 columns * 2 BF16 bytes * K/V.
inline constexpr std::size_t kPageRows = 32;
inline constexpr std::uint64_t kRowPairBytes = 2 * 256 * 2 * 2;
inline constexpr std::uint64_t kPagePairBytes = kPageRows * kRowPairBytes;
static_assert(kPagePairBytes == 65'536);

struct PageAddress {
  std::int32_t slot;
  std::uint64_t generation;
  bool operator==(const PageAddress&) const = default;
};

struct CopySegment {
  PageAddress source;
  PageAddress destination;
  std::size_t rows; // Copy rows [0, rows) of both K and V.
};

struct WriteSegment {
  std::size_t sourceRow; // Row in the appended K/V inputs, not global position.
  PageAddress destination;
  std::size_t destinationRow;
  std::size_t rows;
};

struct PagePoolStats {
  std::size_t capacityPages;
  std::size_t freePages;
  std::size_t reservedPages; // Unpublished leases, possibly pinned by GPU work.
  std::size_t publishedPages;
  std::size_t uniqueLivePages;
  std::size_t peakUniqueLivePages;
  std::uint64_t uniquePhysicalBytes;
  std::uint64_t peakUniquePhysicalBytes;
  std::uint64_t committedPlans;
  std::uint64_t committedAppendedRows;
  std::uint64_t committedPlannedTailCopies;
  std::uint64_t committedPlannedCopiedRows;
  std::uint64_t committedPlannedCopiedBytes;
};

class PagePoolExhausted : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

namespace detail {
struct PoolState;
struct PageListStorage;
struct AppendPlanStorage;
} // namespace detail

// Copies/forks share immutable page leases. No mutating page access is exposed.
class PageList {
 public:
  PageList() = default;
  bool valid() const noexcept;
  std::size_t tokenCount() const;
  std::size_t pageCount() const;
  std::uint64_t logicalBytes() const;
  const std::vector<std::int32_t>& pageIDs() const;
  const std::vector<PageAddress>& pageAddresses() const;
  // Keep this pin in the lazy primitive and GPU completion handler.
  std::shared_ptr<const void> lifetimePin() const;

 private:
  explicit PageList(std::shared_ptr<const detail::PageListStorage> storage);
  std::shared_ptr<const detail::PageListStorage> storage_;
  friend class PagePool;
  friend class AppendTransaction;
  friend struct detail::AppendPlanStorage;
};

// New slots are reserved as one transaction. Every partial tail is copied into
// a new slot, even when no other state is currently visible to the caller.
class AppendTransaction {
 public:
  AppendTransaction(AppendTransaction&&) noexcept;
  AppendTransaction& operator=(AppendTransaction&&) noexcept;
  AppendTransaction(const AppendTransaction&) = delete;
  AppendTransaction& operator=(const AppendTransaction&) = delete;
  ~AppendTransaction();

  std::size_t tokenCount() const;
  const std::vector<std::int32_t>& destinationPageIDs() const;
  const std::vector<PageAddress>& destinationPageAddresses() const;
  const std::vector<CopySegment>& copies() const;
  const std::vector<WriteSegment>& writes() const;
  // Captures source and destination leases. A rolled-back transaction with a
  // live pin keeps its slots reserved until the pin is released.
  std::shared_ptr<const void> lifetimePin() const;
  // Publication only: NOT evidence that any GPU write completed. Exactly once.
  PageList commit();
  // Idempotent. Published lists survive; pins defer physical-slot reclamation.
  void rollback() noexcept;

 private:
  explicit AppendTransaction(std::shared_ptr<detail::AppendPlanStorage> plan);
  const detail::AppendPlanStorage& requirePlan() const;
  std::shared_ptr<detail::AppendPlanStorage> plan_;
  bool committed_ = false;
  friend class PagePool;
};

class PagePool {
 public:
  explicit PagePool(std::size_t capacityPages);
  PageList empty() const;
  AppendTransaction planAppend(const PageList& source,
                               std::size_t appendedRows) const;
  PagePoolStats stats() const;
  // Validate pool identity, live generation and logical/page-count consistency.
  // Useful at an ABI boundary before handing raw integer IDs to a kernel.
  void validate(const PageList& list) const;

 private:
  std::shared_ptr<detail::PoolState> state_;
};

} // namespace anern::paged
