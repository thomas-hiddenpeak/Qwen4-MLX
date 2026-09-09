#pragma once
#include <mlx/array.h>
#include <mlx/stream.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace anemlx::pool {
struct PoolImpl;
struct StateImpl;

struct Stats {
  uint64_t physical_pages = 0;
  uint64_t arena_logical_bytes = 0;
  uint64_t arena_allocated_bytes = 0;
  uint64_t live_pages = 0;
  uint64_t free_pages = 0;
  uint64_t high_water_pages = 0;
  uint64_t encoded_writes = 0;
  uint64_t encoded_reads = 0;
  uint64_t encoded_materializations = 0;
  uint64_t copied_tail_bytes = 0;
  uint64_t written_row_bytes = 0;
  uint64_t materialized_bytes = 0;
  uint64_t in_flight_operations = 0;
  uint64_t completed_operations = 0;
  uint64_t failed_operations = 0;
  uint64_t key_buffer_identity = 0;
  uint64_t value_buffer_identity = 0;
};

class State {
 public:
  int logical_tokens() const;
  std::vector<int32_t> page_ids() const;
  State fork() const;
  State append(const mlx::core::array& keys, const mlx::core::array& values) const;
  mlx::core::array read(const mlx::core::array& query,
      const std::optional<mlx::core::array>& mask = std::nullopt) const;
  std::pair<mlx::core::array, mlx::core::array> materialize() const;
  mlx::core::array ready() const;
 private:
  explicit State(std::shared_ptr<const StateImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<const StateImpl> impl_;
  friend class Pool;
};

class Pool {
 public:
  static std::shared_ptr<Pool> create(int physical_pages, mlx::core::Stream stream);
  State import_kv(const mlx::core::array& keys, const mlx::core::array& values);
  Stats stats() const;
 private:
  explicit Pool(std::shared_ptr<PoolImpl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<PoolImpl> impl_;
};
}  // namespace anemlx::pool
