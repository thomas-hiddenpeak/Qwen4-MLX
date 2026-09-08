// Independent Q24/KV2 layout/boolean-mask probe. Does not include or modify the
// already compiled capacity_probe.cpp; does not construct the learned indexer.
#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>
#include <mlx/version.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
constexpr int query_heads = 24, kv_heads = 2, gqa_factor = 12, dim = 256;
bool async_mode = false;
int checks = 0, cases = 0;

void require(bool value, const std::string& message) {
  ++checks;
  if (!value) throw std::runtime_error(message);
}

mx::Stream gpu() { return mx::default_stream(mx::Device(mx::Device::gpu)); }

void ready(mx::array& a, mx::array& b) {
  if (async_mode) { mx::async_eval(a, b); mx::synchronize(gpu()); }
  else mx::eval(a, b);
  a.wait(); b.wait();
}

std::vector<uint16_t> bits(const mx::array& a) {
  require(a.is_available() && a.ndim() == 4 && a.dtype() == mx::bfloat16,
          "Expected available BF16 rank-four array");
  const auto* data = a.data<uint16_t>();
  std::vector<uint16_t> value; value.reserve(a.size());
  for (int b = 0; b < a.shape(0); ++b)
    for (int h = 0; h < a.shape(1); ++h)
      for (int t = 0; t < a.shape(2); ++t)
        for (int d = 0; d < a.shape(3); ++d)
          value.push_back(data[b*a.strides()[0] + h*a.strides()[1] + t*a.strides()[2] + d*a.strides()[3]]);
  return value;
}

void equal_bits(const mx::array& a, const mx::array& b, const std::string& label) {
  require(a.shape() == b.shape(), label + " shape mismatch");
  const auto lhs = bits(a), rhs = bits(b);
  const auto first = std::mismatch(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
  require(first.first == lhs.end(), label + " BF16 mismatch at element " +
          std::to_string(std::distance(lhs.begin(), first.first)));
}

// All logical values are exactly representable BF16 numbers. Padding is
// deliberately different and finite, so accidental backing-row reads matter.
mx::array values(int heads, int rows, int logical_rows, int salt, bool value_tensor) {
  std::vector<float> host(static_cast<size_t>(heads) * rows * dim);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < rows; ++t)
      for (int d = 0; d < dim; ++d) {
        float x = static_cast<float>((h*71 + t*43 + d*19 + salt) % 509 - 254);
        x /= value_tensor ? 16.0f : 128.0f;
        if (t >= logical_rows) x = value_tensor ? -80.0f-h : 48.0f+h;
        host[(static_cast<size_t>(h)*rows + t)*dim + d] = x;
      }
  return mx::array(host.begin(), {1, heads, rows, dim}, mx::bfloat16);
}

struct Mask {
  std::string label;
  std::vector<uint8_t> visible;
  int selected_blocks = 0, tail_tokens = 0;
};

std::vector<Mask> masks(int rows) {
  std::vector<Mask> result;
  result.push_back({"none", {}});
  result.push_back({"all_true_control", std::vector<uint8_t>(rows, 1)});
  Mask last{"last_only_gqa_mapping_control", std::vector<uint8_t>(rows, 0)};
  last.visible.back() = 1; result.push_back(std::move(last));
  if (rows > 2051) {
    const int blocks = rows/4;
    Mask qsa{"representative_qsa_512_blocks_plus_tail", std::vector<uint8_t>(rows, 0), 512, rows%4};
    int stride = 73;
    while (std::gcd(stride, blocks) != 1) ++stride;
    for (int i = 0; i < 512; ++i) {
      const int block = (i*stride + 17) % blocks;
      for (int lane = 0; lane < 4; ++lane) qsa.visible[block*4 + lane] = 1;
    }
    for (int t = blocks*4; t < rows; ++t) qsa.visible[t] = 1;
    require(std::accumulate(qsa.visible.begin(), qsa.visible.end(), 0) == 2048 + rows%4,
            "Representative QSA mask count is wrong");
    result.push_back(std::move(qsa));
  }
  return result;
}

void print_layout(const mx::array& a) {
  std::cout << "{\"metal_buffer\":\"0x" << std::hex
            << reinterpret_cast<std::uintptr_t>(a.buffer().ptr()) << std::dec
            << "\",\"offset_bytes\":" << a.offset() << ",\"allocation_bytes\":" << a.buffer_size()
            << ",\"logical_bytes\":" << a.nbytes() << ",\"data_elements\":" << a.data_size()
            << ",\"row_contiguous\":" << a.flags().row_contiguous << ",\"strides\":[";
  for (int i = 0; i < 4; ++i) { if (i) std::cout << ','; std::cout << a.strides()[i]; }
  std::cout << "]}";
}

void check_one_length(int rows) {
  // Deliberately leave padding even at exact 256-row boundaries. This is a
  // layout stress fixture, not a proposed change to the production grow rule.
  const int capacity = ((rows + 1 + 255)/256)*256;
  auto backing_k = values(kv_heads, capacity, rows, 11, false);
  auto backing_v = values(kv_heads, capacity, rows, 137, true);
  auto compact_k = values(kv_heads, rows, rows, 11, false);
  auto compact_v = values(kv_heads, rows, rows, 137, true);
  auto view_k = mx::slice(backing_k, {0,0,0,0}, {1,kv_heads,rows,dim}, gpu());
  auto view_v = mx::slice(backing_v, {0,0,0,0}, {1,kv_heads,rows,dim}, gpu());
  ready(view_k, view_v);
  require(query_heads / kv_heads == gqa_factor, "Incorrect GQA fixture");
  require(view_k.strides()[1] == static_cast<int64_t>(capacity)*dim && view_k.strides()[3] == 1,
          "K prefix does not have the intended capacity head stride");
  require(view_v.strides()[1] == static_cast<int64_t>(capacity)*dim && view_v.strides()[3] == 1,
          "V prefix does not have the intended capacity head stride");
  require(!view_k.flags().row_contiguous && !view_v.flags().row_contiguous,
          "Fixture accidentally became contiguous");
  require(view_k.buffer().ptr() == backing_k.buffer().ptr() && view_v.buffer().ptr() == backing_v.buffer().ptr(),
          "Prefix view unexpectedly materialized before SDPA");
  require(compact_k.buffer().ptr() != backing_k.buffer().ptr() && compact_v.buffer().ptr() != backing_v.buffer().ptr(),
          "Compact reference must have independent storage");
  equal_bits(view_k, compact_k, "K prefix inputs"); equal_bits(view_v, compact_v, "V prefix inputs");
  const auto old_k = bits(backing_k), old_v = bits(backing_v);
  const auto key_identity = backing_k.buffer().ptr(), value_identity = backing_v.buffer().ptr();
  auto query = values(query_heads, 1, 1, 211, false);
  for (const auto& item : masks(rows)) {
    std::optional<mx::array> mask;
    if (!item.visible.empty()) {
      mask.emplace(item.visible.begin(), mx::Shape{1,1,1,rows}, mx::bool_);
      require(mask->dtype() == mx::bool_ && mask->shape() == mx::Shape({1,1,1,rows}), "Wrong bool mask shape");
    }
    // Same mask, logical T, query and default force_fused=false as AR decode.
    // Actual kernel selection remains the installed MLX implementation's job.
    auto lhs = mx::fast::scaled_dot_product_attention(query, view_k, view_v,
        1.0f/16.0f, mask ? "array" : "", mask, {}, false, gpu());
    auto rhs = mx::fast::scaled_dot_product_attention(query, compact_k, compact_v,
        1.0f/16.0f, mask ? "array" : "", mask, {}, false, gpu());
    ready(lhs, rhs);
    require(lhs.shape() == mx::Shape({1,query_heads,1,dim}), "GQA output shape is wrong");
    equal_bits(lhs, rhs, "GQA noncontiguous prefix versus compact");
    const auto output = bits(lhs);
    for (auto value : output) require(std::isfinite(std::bit_cast<float>(static_cast<uint32_t>(value)<<16)), "Nonfinite GQA output");
    if (item.label == "last_only_gqa_mapping_control") {
      const auto vv = bits(compact_v);
      for (int h = 0; h < query_heads; ++h)
        for (int d = 0; d < dim; ++d)
          require(output[h*dim+d] == vv[(static_cast<size_t>(h/gqa_factor)*rows + rows-1)*dim+d],
                  "Last-only mask did not select the matching KV head's final row");
    }
    ++cases;
    std::cout << "{\"event\":\"gqa_case\",\"passed\":true,\"rows\":" << rows
              << ",\"capacity\":" << capacity << ",\"q_heads\":24,\"kv_heads\":2,\"gqa_factor\":12"
              << ",\"query_rows\":1,\"mask\":" << std::quoted(item.label)
              << ",\"visible_tokens\":" << (item.visible.empty() ? rows : std::accumulate(item.visible.begin(),item.visible.end(),0))
              << ",\"selected_complete_blocks\":" << item.selected_blocks << ",\"tail_tokens\":" << item.tail_tokens
              << ",\"force_fused\":false,\"k_view\":";
    print_layout(view_k); std::cout << ",\"k_compact\":"; print_layout(compact_k);
    std::cout << "}\n" << std::flush;
  }
  require(backing_k.buffer().ptr() == key_identity && backing_v.buffer().ptr() == value_identity,
          "Read-only SDPA changed backing identity");
  require(bits(backing_k) == old_k && bits(backing_v) == old_v, "Read-only SDPA modified K/V or padding");
}

int main(int argc, char** argv) {
  std::cout << std::boolalpha << std::setprecision(17);
  try {
    bool requested = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--sdpa") requested = true;
      else if (arg == "--async-eval") async_mode = true;
      else throw std::invalid_argument("options: --sdpa [--async-eval]");
    }
    if (!requested) { std::cerr << "Explicit --sdpa is required; no mechanism was tested.\n"; return 2; }
    std::cout << "{\"event\":\"start\",\"mlx_version\":" << std::quoted(mx::version())
              << ",\"scope\":\"GQA capacity-view numerics with representative bool masks\",\"model_loaded\":false"
              << ",\"actual_qsa_indexer_executed\":false,\"async_eval\":" << async_mode << "}\n" << std::flush;
    for (int rows : {255,256,257,1023,1024,1025,2051,2052,2053,4095,4096,4097,11232,11233}) check_one_length(rows);
    mx::synchronize(gpu());
    require(cases == 49, "Unexpected GQA case coverage");
    std::cout << "{\"event\":\"summary\",\"passed\":true,\"lengths\":14,\"cases\":" << cases
              << ",\"checks\":" << checks << ",\"actual_qsa_indexer_executed\":false}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "GQA capacity probe failed: " << error.what() << '\n';
    std::cout << "{\"event\":\"summary\",\"passed\":false,\"cases\":" << cases << ",\"checks\":" << checks << "}\n";
    return 1;
  }
}
