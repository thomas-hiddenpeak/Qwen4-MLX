// Dispatch geometry follows local MLX scaled_dot_product_attention.cpp,
// Copyright 2024 Apple Inc. (MIT, LICENSE.MLX.txt). No installed code is changed.
#include "paged_sdpa_reader.h"
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

#ifndef ANEMLX_PAGED_METALLIB
#error "Provide the absolute ANEMLX_PAGED_METALLIB path at compile time"
#endif
#ifndef ANEMLX_PAGED_LIBRARY_ID
#error "Provide a distinct ANEMLX_PAGED_LIBRARY_ID for this candidate"
#endif

namespace anemlx::paged {
namespace mx = mlx::core;
namespace {
constexpr int max_token_bound = 131072;
std::atomic<uint64_t> encoded_read_count{0};
void require(bool ok, const char* message) {
  if (!ok) throw std::invalid_argument(message);
}
std::vector<int32_t> checked_pages(std::vector<int32_t> pages, int physical, int n) {
  require(n > 0 && n <= max_token_bound, "Logical tokens outside 1...131072");
  require(physical > 0 && physical <= 2 * (max_token_bound / page_tokens), "Invalid physical page bound");
  require(pages.size() == size_t((n + page_tokens - 1) / page_tokens), "Wrong logical page count");
  for (int32_t p : pages) require(p >= 0 && p < physical, "Physical page ID out of bounds");
  return pages;
}
struct Strides { size_t head, token, page; };
Strides storage_strides(const mx::array& a, StorageKind kind, int physical, int n) {
  require(a.ndim() == 4 && a.dtype() == mx::bfloat16 && a.shape(1) == kv_heads &&
              a.shape(3) == head_dim && a.strides(3) == 1,
          "Reader requires BF16 rank-four K/V with unit D stride");
  if (kind == StorageKind::PageMajor) {
    require(a.shape(0) == physical && a.shape(2) == page_tokens,
            "Page-major K/V shape does not match physical table");
  } else {
    require(a.shape(0) == 1 && a.shape(2) == n && physical == (n + 31) / 32,
            "Head-major K/V must expose exactly the logical prefix");
  }
  require(a.strides(1) > 0 && a.strides(2) > 0 && a.strides(0) > 0,
          "K/V strides must be positive");
  return {size_t(a.strides(1)), size_t(a.strides(2)),
          kind == StorageKind::PageMajor ? size_t(a.strides(0)) : size_t(32 * a.strides(2))};
}
void require_group(MTL::ComputePipelineState* kernel, MTL::Size group) {
  require(group.width * group.height * group.depth <= kernel->maxTotalThreadsPerThreadgroup(),
          "Device cannot execute the pinned MLX threadgroup geometry");
}
class Reader final : public mx::UnaryPrimitive {
 public:
  Reader(mx::Stream stream, int n, int physical, StorageKind kind, bool mask)
      : UnaryPrimitive(stream), n_(n), physical_(physical), kind_(kind), mask_(mask) {}
  const char* name() const override { return "ANEMLXPagedSDPAVector32"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Paged SDPA reader is GPU-only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& output) override {
    const auto& q = inputs[0]; const auto& k = inputs[1]; const auto& v = inputs[2];
    require(q.flags().row_contiguous && inputs[3].flags().row_contiguous,
            "Evaluated query/page-table layout must be row-contiguous");
    const auto ks = storage_strides(k, kind_, physical_, n_);
    const auto vs = storage_strides(v, kind_, physical_, n_);
    if (mask_) require(inputs[4].strides(3) >= 0 && inputs[4].strides(3) <= std::numeric_limits<int32_t>::max(),
                       "Mask token stride must fit nonnegative int32");
    const auto plan = dispatch_info(n_, stream());
    auto& d = mx::metal::device(stream().device);
    auto* library = d.get_library(ANEMLX_PAGED_LIBRARY_ID, ANEMLX_PAGED_METALLIB);
    std::string name = plan.two_pass ? "anemlx_paged_sdpa_vector_2pass_1_bfloat16_t_256_256"
                                     : "anemlx_paged_sdpa_vector_bfloat16_t_256_256";
    const bool has_mask = mask_, bool_mask = mask_, qt = false, causal = false,
               float_mask = false, sinks = false;
    const int blocks = plan.blocks;
    mx::metal::MTLFCList constants = {
        {&has_mask, MTL::DataType::DataTypeBool, 20},
        {&qt, MTL::DataType::DataTypeBool, 21},
        {&causal, MTL::DataType::DataTypeBool, 22},
        {&bool_mask, MTL::DataType::DataTypeBool, 23},
        {&float_mask, MTL::DataType::DataTypeBool, 24},
        {&sinks, MTL::DataType::DataTypeBool, 25}};
    if (plan.two_pass) constants.emplace_back(&blocks, MTL::DataType::DataTypeInt, 26);
    const std::string hash = name + (mask_ ? "_bool_" : "_none_") + std::to_string(blocks);
    auto* kernel = d.get_kernel(name, library, hash, constants);
    const MTL::Size group = plan.two_pass ? MTL::Size(32, 12, 1) : MTL::Size(1024, 1, 1);
    require_group(kernel, group);
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& enc = mx::metal::get_command_encoder(stream());
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(q, 0); enc.set_input_array(k, 1); enc.set_input_array(v, 2);
    enc.set_input_array(inputs[3], 19);
    enc.set_bytes(ks.page, 20); enc.set_bytes(vs.page, 21);
    const float scale = 1.0f / 16.0f;
    if (!plan.two_pass) {
      const int gqa = 12;
      enc.set_output_array(output, 3); enc.set_bytes(gqa, 4); enc.set_bytes(n_, 5);
      enc.set_bytes(ks.head, 6); enc.set_bytes(ks.token, 7);
      enc.set_bytes(vs.head, 8); enc.set_bytes(vs.token, 9); enc.set_bytes(scale, 10);
      if (mask_) bind_mask(enc, inputs[4], 11, 13);
      enc.dispatch_threadgroups(MTL::Size(24, 1, 1), group);
      encoded_read_count.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    // Exactly the local MLX partial layout and BF16 intermediate quantization.
    // The encoder owns temporary arrays until real GPU completion; metadata
    // lifetime or host return does not release these buffers prematurely.
    auto temp = [&enc](mx::Shape shape, mx::Dtype type) {
      mx::array a(std::move(shape), type, nullptr, {});
      a.set_data(mx::allocator::malloc(a.nbytes())); enc.add_temporary(a); return a;
    };
    auto partial = temp({1, 24, 1, blocks, 256}, mx::bfloat16);
    auto sums = temp({1, 24, 1, blocks}, mx::float32);
    auto maxs = temp({1, 24, 1, blocks}, mx::float32);
    enc.set_output_array(partial, 3); enc.set_output_array(sums, 4); enc.set_output_array(maxs, 5);
    enc.set_bytes(n_, 7); enc.set_bytes(ks.head, 8); enc.set_bytes(ks.token, 9);
    enc.set_bytes(vs.head, 10); enc.set_bytes(vs.token, 11); enc.set_bytes(scale, 12);
    if (mask_) bind_mask(enc, inputs[4], 13, 15);
    enc.dispatch_threadgroups(MTL::Size(2, 1, blocks), group);
    // Use the installed, unmodified second-pass reduction. Registering output
    // then input arrays lets the shared encoder track the dependency barrier.
    kernel = d.get_kernel("sdpa_vector_2pass_2_bfloat16_t_256");
    require_group(kernel, MTL::Size(1024, 1, 1));
    enc.set_compute_pipeline_state(kernel);
    enc.set_input_array(partial, 0); enc.set_input_array(sums, 1); enc.set_input_array(maxs, 2);
    enc.set_output_array(output, 3); enc.set_bytes(blocks, 4);
    enc.dispatch_threadgroups(MTL::Size(24, 1, 1), MTL::Size(1024, 1, 1));
    encoded_read_count.fetch_add(1, std::memory_order_relaxed);
  }
 private:
  static void bind_mask(mx::metal::CommandEncoder& enc, const mx::array& mask, int slot, int strides) {
    enc.set_input_array(mask, slot);
    const int32_t token = mask.shape(3) > 1 ? int32_t(mask.strides(3)) : 0, zero = 0;
    enc.set_bytes(token, strides); enc.set_bytes(zero, strides + 1); enc.set_bytes(zero, strides + 2);
  }
  int n_, physical_; StorageKind kind_; bool mask_;
};
}  // namespace

uint64_t encoded_reads() { return encoded_read_count.load(std::memory_order_relaxed); }

IdentityPageTable::IdentityPageTable(int maximum)
    : max_tokens_(maximum), device_pages_([maximum]() {
        require(maximum > 0 && maximum <= max_token_bound, "Identity maximum outside 1...131072");
        std::vector<int32_t> ids((maximum + 31) / 32);
        std::iota(ids.begin(), ids.end(), 0);
        return mx::array(ids.begin(), mx::Shape{int(ids.size())}, mx::int32);
      }()) {}

PageTable::PageTable(std::vector<int32_t> pages, int physical_pages, int logical_tokens)
    : pages_(checked_pages(std::move(pages), physical_pages, logical_tokens)),
      physical_pages_(physical_pages), tokens_(logical_tokens),
      device_pages_(pages_.begin(), mx::Shape{int(pages_.size())}, mx::int32) {}

DispatchInfo dispatch_info(int n, mx::Stream stream) {
  require(stream.device.type == mx::Device::gpu && n > 0 && n <= max_token_bound, "Invalid reader stream/length");
  auto& d = mx::metal::device(stream.device);
  require(!d.get_architecture().empty(), "Unknown Metal architecture");
  const char devc = d.get_architecture().back();
  const bool two = ((devc == 'd' || devc == 's') && n >= 1024) || n >= 4096;
  if (!two) return {false, 0, 0};
  // Exact local policy for n_simds = QH/KVH * S = 12, including the local
  // override's multiple-of-32 rule. A bounded candidate rejects absurd values.
  int blocks = 64;
  if (devc == 's') {
    if (n > 1024) blocks = n <= 8192 ? 128 : n <= 32768 ? 256 : n <= 65536 ? 512 : 1024;
  } else if (devc == 'd') {
    blocks = n >= 65536 ? 1024 : n >= 16384 ? 512 : 128;
  }
  if (const char* env = std::getenv("MLX_SDPA_BLOCKS")) {
    char* end = nullptr; const long value = std::strtol(env, &end, 10);
    require(end != env && *end == '\0' && value >= 0 && value <= 4096, "Invalid/beyond-bound MLX_SDPA_BLOCKS");
    if (value > 0) blocks = int((value + 31) / 32) * 32;
  }
  return {true, blocks, uint64_t(24) * blocks * (256 * 2 + 2 * 4)};
}

mx::array read(const mx::array& q, const mx::array& k, const mx::array& v,
               const PageTable& table, StorageKind kind, const std::optional<mx::array>& mask,
               mx::Stream stream) {
  require(stream.device.type == mx::Device::gpu, "Reader needs an explicit GPU stream");
  require(q.shape() == mx::Shape({1, 24, 1, 256}) && q.dtype() == mx::bfloat16, "Invalid query shape/dtype");
  require(kind == StorageKind::PageMajor || kind == StorageKind::HeadMajor, "Invalid storage kind");
  // Strides of lazy tensors are checked only at eval. Shapes are stable now.
  require(k.ndim() == 4 && v.ndim() == 4 && k.dtype() == mx::bfloat16 && v.dtype() == mx::bfloat16,
          "Invalid key/value dtype or rank");
  if (kind == StorageKind::HeadMajor) {
    for (size_t logical_page = 0; logical_page < table.pages_.size(); ++logical_page) {
      const int rows = std::min(32, table.tokens_ - int(logical_page) * 32);
      require(table.pages_[logical_page] * 32 + rows <= table.tokens_, "Head-major page map reads beyond its logical view");
    }
  }
  std::vector<mx::array> inputs{q, k, v, table.device_pages_};
  if (mask) {
    require(mask->dtype() == mx::bool_ && mask->shape() == mx::Shape({1, 1, 1, table.tokens_}),
            "Expected logical boolean QSA mask [1,1,1,T]");
    inputs.push_back(*mask);
  }
  return mx::array({1, 24, 1, 256}, mx::bfloat16,
      std::make_shared<Reader>(stream, table.tokens_, table.physical_pages_, kind, mask.has_value()),
      std::move(inputs));
}

mx::array read_identity(const mx::array& q, const mx::array& k, const mx::array& v,
                        const IdentityPageTable& table, const std::optional<mx::array>& mask,
                        mx::Stream stream) {
  require(stream.device.type == mx::Device::gpu, "Reader needs an explicit GPU stream");
  require(q.shape() == mx::Shape({1, 24, 1, 256}) && q.dtype() == mx::bfloat16, "Invalid query shape/dtype");
  require(k.ndim() == 4 && v.ndim() == 4 && k.dtype() == mx::bfloat16 && v.dtype() == mx::bfloat16,
          "Invalid key/value dtype or rank");
  const int n = k.shape(2);
  require(n > 0 && n <= table.max_tokens_, "Identity reader exceeds initialized length bound");
  require(k.shape() == mx::Shape({1, 2, n, 256}) && v.shape() == k.shape(), "Identity reader requires head-major prefix views");
  std::vector<mx::array> inputs{q, k, v, table.device_pages_};
  if (mask) {
    require(mask->dtype() == mx::bool_ && mask->shape() == mx::Shape({1, 1, 1, n}), "Expected logical boolean QSA mask [1,1,1,T]");
    inputs.push_back(*mask);
  }
  return mx::array({1, 24, 1, 256}, mx::bfloat16,
      std::make_shared<Reader>(stream, n, (n + 31) / 32, StorageKind::HeadMajor, mask.has_value()),
      std::move(inputs));
}
}  // namespace anemlx::paged
