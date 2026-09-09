#pragma once
#include <mlx/array.h>
#include <mlx/stream.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace anemlx::paged {
constexpr int page_tokens = 32, query_heads = 24, kv_heads = 2, head_dim = 256;
enum class StorageKind { PageMajor, HeadMajor };
struct ReaderLifetime {
  std::shared_ptr<const void> owner;
  std::optional<mlx::core::array> dependency;
  std::function<void()> validate;
  std::function<void()> submitted;
  std::function<void()> encoded;
  std::function<void(bool)> observed_status;
  std::function<void(bool)> completed;
};

// Construct once from small scheduler metadata; no tensor readback in reader.
// The immutable page IDs are copied into a private MLX graph input. Inputs are
// read-only: callers must keep physical pages immutable until eval completes.
class PageTable {
 public:
  PageTable(std::vector<int32_t> pages, int physical_pages, int logical_tokens);
  int logical_tokens() const { return tokens_; }
  int physical_pages() const { return physical_pages_; }
  const std::vector<int32_t>& page_ids() const { return pages_; }

 private:
  std::vector<int32_t> pages_;
  int physical_pages_, tokens_;
  mlx::core::array device_pages_;
  friend mlx::core::array read(
      const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
      const PageTable&, StorageKind, const std::optional<mlx::core::array>&,
      mlx::core::Stream, ReaderLifetime);
};

struct DispatchInfo {
  bool two_pass;
  int blocks;
  uint64_t logical_scratch_bytes;
};
DispatchInfo dispatch_info(int tokens, mlx::core::Stream stream);
// Successful full read encodings only; neither graph construction nor GPU
// completion. Process-wide monotonic counter, never reset while graphs live.
uint64_t encoded_reads();

class IdentityPageTable {
 public:
  explicit IdentityPageTable(int max_tokens);
  int max_tokens() const { return max_tokens_; }
  uint64_t metadata_bytes() const { return uint64_t((max_tokens_ + 31) / 32) * 4; }
 private:
  int max_tokens_;
  mlx::core::array device_pages_;
  friend mlx::core::array read_identity(
      const mlx::core::array&, const mlx::core::array&, const mlx::core::array&,
      const IdentityPageTable&, const std::optional<mlx::core::array>&,
      mlx::core::Stream);
};

// Head-major fast path: one identity table per runtime, no per-token table
// rebuild, host scan, array slice, or K/V conversion. T is keys.shape(2).
mlx::core::array read_identity(
    const mlx::core::array& query, const mlx::core::array& keys,
    const mlx::core::array& values, const IdentityPageTable& table,
    const std::optional<mlx::core::array>& mask, mlx::core::Stream stream);

// Fixed BF16 Q[1,24,1,256], output same shape, scale=1/16.
// PageMajor: K/V[P,2,32,256]. HeadMajor: K/V[1,2,T,256], including capacity
// views with head stride C*256 and C not divisible by 32. Both share one kernel
// with independent actual K/V head/page/token strides. No K/V pack or gather.
// Optional boolean QSA visibility [1,1,1,T] is indexed by LOGICAL token position.
mlx::core::array read(
    const mlx::core::array& query,
    const mlx::core::array& keys,
    const mlx::core::array& values,
    const PageTable& table,
    StorageKind storage,
    const std::optional<mlx::core::array>& mask,
    mlx::core::Stream stream, ReaderLifetime lifetime = {});
}  // namespace anemlx::paged
