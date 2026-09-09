#include "immutable_page_pool.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>

namespace anern::paged {
namespace detail {

enum class SlotStatus { free, reserved, published };
struct Slot { SlotStatus status = SlotStatus::free; std::uint64_t generation = 0; };

struct PoolState {
  explicit PoolState(std::size_t count) : slots(count) {
    freeSlots.reserve(count);
    for (std::size_t i = count; i > 0; --i)
      freeSlots.push_back(static_cast<std::int32_t>(i - 1));
  }
  mutable std::mutex mutex;
  std::vector<Slot> slots;
  std::vector<std::int32_t> freeSlots;
  std::size_t reserved = 0, published = 0, peak = 0;
  std::uint64_t plans = 0, appended = 0, copies = 0, copiedRows = 0;
};

struct PageLease {
  explicit PageLease(std::shared_ptr<PoolState> owner) : owner(std::move(owner)) {}
  ~PageLease() {
    if (!active) return;
    std::lock_guard lock(owner->mutex);
    auto& slot = owner->slots[static_cast<std::size_t>(address.slot)];
    // A live lease is the sole reclaim authority for this generation.
    if (slot.generation != address.generation || slot.status == SlotStatus::free)
      std::terminate();
    if (slot.status == SlotStatus::reserved) --owner->reserved;
    else --owner->published;
    slot.status = SlotStatus::free;
    owner->freeSlots.push_back(address.slot); // Capacity was reserved once.
  }
  std::shared_ptr<PoolState> owner;
  PageAddress address{-1, 0};
  bool active = false;
};

struct PageListStorage {
  std::shared_ptr<PoolState> owner;
  std::size_t tokens = 0;
  std::vector<std::shared_ptr<PageLease>> pages;
  std::vector<std::int32_t> ids;
  std::vector<PageAddress> addresses;
};

struct AppendPlanStorage {
  PageList source, destination;
  std::vector<std::shared_ptr<PageLease>> reserved;
  std::vector<CopySegment> copies;
  std::vector<WriteSegment> writes;
  std::size_t appendedRows = 0;
};

static std::vector<std::shared_ptr<PageLease>> reserve(
    const std::shared_ptr<PoolState>& owner, std::size_t count) {
  std::vector<std::shared_ptr<PageLease>> result;
  result.reserve(count);
  // Allocate handles before touching pool state: allocation failure is atomic.
  for (std::size_t i = 0; i < count; ++i)
    result.push_back(std::make_shared<PageLease>(owner));
  std::lock_guard lock(owner->mutex);
  if (count > owner->freeSlots.size())
    throw PagePoolExhausted("insufficient free KV page slots");
  for (std::size_t i = 0; i < count; ++i) {
    const auto id = owner->freeSlots[owner->freeSlots.size() - 1 - i];
    if (owner->slots[static_cast<std::size_t>(id)].generation ==
        std::numeric_limits<std::uint64_t>::max())
      throw std::overflow_error("KV page generation exhausted");
  }
  for (auto& lease : result) {
    const auto id = owner->freeSlots.back();
    owner->freeSlots.pop_back();
    auto& slot = owner->slots[static_cast<std::size_t>(id)];
    slot.status = SlotStatus::reserved;
    ++slot.generation;
    lease->address = {id, slot.generation};
    lease->active = true;
  }
  owner->reserved += count;
  owner->peak = std::max(owner->peak, owner->reserved + owner->published);
  return result;
}

static const PageListStorage& requireList(
    const std::shared_ptr<const PageListStorage>& storage) {
  if (!storage) throw std::invalid_argument("unbound KV page list");
  return *storage;
}

static void saturatingAdd(std::uint64_t& target, std::uint64_t value) noexcept {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  target = value > maximum - target ? maximum : target + value;
}

} // namespace detail

PageList::PageList(std::shared_ptr<const detail::PageListStorage> storage)
    : storage_(std::move(storage)) {}
bool PageList::valid() const noexcept { return bool(storage_); }
std::size_t PageList::tokenCount() const { return detail::requireList(storage_).tokens; }
std::size_t PageList::pageCount() const { return detail::requireList(storage_).pages.size(); }
std::uint64_t PageList::logicalBytes() const { return tokenCount() * kRowPairBytes; }
const std::vector<std::int32_t>& PageList::pageIDs() const {
  return detail::requireList(storage_).ids;
}
const std::vector<PageAddress>& PageList::pageAddresses() const {
  return detail::requireList(storage_).addresses;
}
std::shared_ptr<const void> PageList::lifetimePin() const {
  (void)detail::requireList(storage_);
  return storage_;
}

AppendTransaction::AppendTransaction(std::shared_ptr<detail::AppendPlanStorage> plan)
    : plan_(std::move(plan)) {}
AppendTransaction::AppendTransaction(AppendTransaction&&) noexcept = default;
AppendTransaction& AppendTransaction::operator=(AppendTransaction&&) noexcept = default;
AppendTransaction::~AppendTransaction() = default;
const detail::AppendPlanStorage& AppendTransaction::requirePlan() const {
  if (!plan_) throw std::logic_error("consumed KV append transaction");
  return *plan_;
}
std::size_t AppendTransaction::tokenCount() const {
  return requirePlan().destination.tokenCount();
}
const std::vector<std::int32_t>& AppendTransaction::destinationPageIDs() const {
  return requirePlan().destination.pageIDs();
}
const std::vector<PageAddress>& AppendTransaction::destinationPageAddresses() const {
  return requirePlan().destination.pageAddresses();
}
const std::vector<CopySegment>& AppendTransaction::copies() const { return requirePlan().copies; }
const std::vector<WriteSegment>& AppendTransaction::writes() const { return requirePlan().writes; }
std::shared_ptr<const void> AppendTransaction::lifetimePin() const {
  (void)requirePlan();
  return plan_;
}
PageList AppendTransaction::commit() {
  const auto& plan = requirePlan();
  if (committed_) throw std::logic_error("KV append transaction already committed");
  auto owner = plan.destination.storage_->owner;
  std::lock_guard lock(owner->mutex);
  for (const auto& lease : plan.reserved) {
    const auto& slot = owner->slots[static_cast<std::size_t>(lease->address.slot)];
    if (lease->owner != owner || slot.generation != lease->address.generation ||
        slot.status != detail::SlotStatus::reserved)
      throw std::logic_error("invalid or stale KV append reservation");
  }
  for (const auto& lease : plan.reserved)
    owner->slots[static_cast<std::size_t>(lease->address.slot)].status =
        detail::SlotStatus::published;
  owner->reserved -= plan.reserved.size();
  owner->published += plan.reserved.size();
  detail::saturatingAdd(owner->plans, 1);
  detail::saturatingAdd(owner->appended, plan.appendedRows);
  detail::saturatingAdd(owner->copies, plan.copies.size());
  for (const auto& copy : plan.copies) detail::saturatingAdd(owner->copiedRows, copy.rows);
  committed_ = true;
  return plan.destination;
}
void AppendTransaction::rollback() noexcept { plan_.reset(); }

PagePool::PagePool(std::size_t capacityPages) {
  if (capacityPages == 0 || capacityPages > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("KV page capacity must fit positive int32");
  state_ = std::make_shared<detail::PoolState>(capacityPages);
}
PageList PagePool::empty() const {
  auto storage = std::make_shared<detail::PageListStorage>();
  storage->owner = state_;
  return PageList(std::move(storage));
}
void PagePool::validate(const PageList& list) const {
  const auto& storage = detail::requireList(list.storage_);
  if (storage.owner != state_) throw std::invalid_argument("KV page list belongs to another pool");
  const auto expectedPages = storage.tokens / kPageRows + (storage.tokens % kPageRows != 0);
  if (storage.pages.size() != expectedPages || storage.ids.size() != expectedPages ||
      storage.addresses.size() != expectedPages)
    throw std::logic_error("inconsistent immutable KV page list");
  std::lock_guard lock(state_->mutex);
  for (std::size_t i = 0; i < storage.pages.size(); ++i) {
    const auto& lease = storage.pages[i];
    if (!lease || lease->owner != state_ || !lease->active || lease->address.slot < 0 ||
        static_cast<std::size_t>(lease->address.slot) >= state_->slots.size())
      throw std::logic_error("invalid KV page lease");
    const auto& slot = state_->slots[static_cast<std::size_t>(lease->address.slot)];
    if (slot.status != detail::SlotStatus::published ||
        slot.generation != lease->address.generation ||
        storage.ids[i] != lease->address.slot || storage.addresses[i] != lease->address)
      throw std::logic_error("stale or unpublished KV page lease");
  }
}
AppendTransaction PagePool::planAppend(const PageList& source, std::size_t appendedRows) const {
  validate(source);
  if (appendedRows == 0) throw std::invalid_argument("KV append must contain rows");
  const auto oldTokens = source.tokenCount();
  const auto maxTokens = state_->slots.size() * kPageRows;
  if (appendedRows > maxTokens || oldTokens > maxTokens - appendedRows)
    throw PagePoolExhausted("logical KV page list exceeds pool capacity");
  const auto total = oldTokens + appendedRows;
  const auto fullPages = oldTokens / kPageRows;
  const auto tailRows = oldTokens % kPageRows;
  const auto newCount = (tailRows + appendedRows) / kPageRows +
                        ((tailRows + appendedRows) % kPageRows != 0);
  auto reserved = detail::reserve(state_, newCount);
  auto destination = std::make_shared<detail::PageListStorage>();
  destination->owner = state_;
  destination->tokens = total;
  destination->pages.reserve(fullPages + newCount);
  destination->pages.insert(destination->pages.end(), source.storage_->pages.begin(),
                            source.storage_->pages.begin() + fullPages);
  destination->pages.insert(destination->pages.end(), reserved.begin(), reserved.end());
  destination->ids.reserve(destination->pages.size());
  destination->addresses.reserve(destination->pages.size());
  for (const auto& lease : destination->pages) {
    destination->ids.push_back(lease->address.slot);
    destination->addresses.push_back(lease->address);
  }
  auto plan = std::make_shared<detail::AppendPlanStorage>();
  plan->source = source;
  plan->destination = PageList(std::move(destination));
  plan->reserved = std::move(reserved);
  plan->appendedRows = appendedRows;
  if (tailRows != 0)
    plan->copies.push_back({source.pageAddresses().back(), plan->reserved.front()->address, tailRows});
  std::size_t inputRow = 0;
  for (std::size_t i = 0; i < plan->reserved.size(); ++i) {
    const auto destinationRow = i == 0 ? tailRows : 0;
    const auto rows = std::min(kPageRows - destinationRow, appendedRows - inputRow);
    plan->writes.push_back({inputRow, plan->reserved[i]->address, destinationRow, rows});
    inputRow += rows;
  }
  return AppendTransaction(std::move(plan));
}
PagePoolStats PagePool::stats() const {
  std::lock_guard lock(state_->mutex);
  const auto live = state_->reserved + state_->published;
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  const auto copiedBytes = state_->copiedRows > maximum / kRowPairBytes ? maximum :
                          state_->copiedRows * kRowPairBytes;
  return {state_->slots.size(), state_->freeSlots.size(), state_->reserved, state_->published,
          live, state_->peak, live * kPagePairBytes, state_->peak * kPagePairBytes,
          state_->plans, state_->appended, state_->copies, state_->copiedRows, copiedBytes};
}

} // namespace anern::paged
