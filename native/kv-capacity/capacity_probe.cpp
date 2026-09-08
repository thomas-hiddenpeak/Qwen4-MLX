// Independent mechanism probe. No model, private memory writes or SDK patches.
#include <mlx/array.h>
#include <mlx/fast.h>
#include <mlx/memory.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>
#include <mlx/version.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace mx = mlx::core;
constexpr int heads = 2, dims = 256, quantum = 256;
struct Pair { mx::array k, v; };
struct State { Pair backing; int rows, capacity, limit; };
struct Native {
  std::uintptr_t metal_buffer;
  int64_t offset;
  size_t allocation_bytes, logical_bytes, data_elements;
  bool contiguous, row_contiguous, donatable;
};
bool async_mode = false, check_sdpa = false;
int checks = 0, append_count = 0, grow_count = 0, cow_count = 0;

void require(bool condition, const std::string& message) {
  ++checks;
  if (!condition) throw std::runtime_error(message);
}

mx::Stream gpu() { return mx::default_stream(mx::Device(mx::Device::gpu)); }

void ready(Pair& pair) {
  if (async_mode) {
    mx::async_eval(pair.k, pair.v);
    mx::synchronize(gpu());
  } else {
    mx::eval(pair.k, pair.v);
  }
  pair.k.wait(); pair.v.wait();
}

// Returning scalar integers cannot increment ArrayDesc/Data/MTL refcounts.
Native native(const mx::array& array) {
  require(array.is_available(), "native identity requires completed evaluation");
  return {reinterpret_cast<std::uintptr_t>(array.buffer().ptr()), array.offset(),
          array.buffer_size(), array.nbytes(), array.data_size(),
          array.flags().contiguous, array.flags().row_contiguous, array.is_donatable()};
}

bool same_allocation(const Native& a, const Native& b) {
  return a.metal_buffer == b.metal_buffer && a.offset == b.offset &&
         a.allocation_bytes == b.allocation_bytes;
}

void print_native(const Native& n) {
  std::cout << "{\"metal_buffer\":\"0x" << std::hex << n.metal_buffer << std::dec
            << "\",\"offset_bytes\":" << n.offset << ",\"allocation_bytes\":" << n.allocation_bytes
            << ",\"logical_bytes\":" << n.logical_bytes << ",\"data_elements\":" << n.data_elements
            << ",\"contiguous\":" << n.contiguous << ",\"row_contiguous\":" << n.row_contiguous
            << ",\"donatable_at_observation\":" << n.donatable << "}";
}

mx::array pattern(int start, int count, int salt) {
  std::vector<float> host(static_cast<size_t>(heads) * count * dims);
  for (int h = 0; h < heads; ++h)
    for (int t = 0; t < count; ++t)
      for (int d = 0; d < dims; ++d) {
        int value = ((start + t) * 43 + h * 71 + d * 19 + salt) % 509 - 254;
        host[(static_cast<size_t>(h) * count + t) * dims + d] = value / 16.0f;
      }
  // Iterator constructor converts/copies the finite host values to BF16.
  // It does not wrap host.data() in a borrowed external buffer.
  return mx::array(host.begin(), {1, heads, count, dims}, mx::bfloat16);
}

Pair patterns(int start, int count) { return {pattern(start, count, 11), pattern(start, count, 137)}; }

Pair logical_view(const Pair& backing, int rows) {
  return {mx::slice(backing.k, {0, 0, 0, 0}, {1, heads, rows, dims}, gpu()),
          mx::slice(backing.v, {0, 0, 0, 0}, {1, heads, rows, dims}, gpu())};
}

std::vector<uint16_t> bits(const mx::array& a) {
  require(a.is_available() && a.dtype() == mx::bfloat16 && a.ndim() == 4,
          "BF16 readback requires available rank-four array");
  const auto* data = a.data<uint16_t>(); // Current installed Metal uses shared storage.
  std::vector<uint16_t> result;
  result.reserve(a.size());
  for (int b = 0; b < a.shape(0); ++b)
    for (int h = 0; h < a.shape(1); ++h)
      for (int t = 0; t < a.shape(2); ++t)
        for (int d = 0; d < a.shape(3); ++d)
          result.push_back(data[b*a.strides()[0] + h*a.strides()[1] + t*a.strides()[2] + d*a.strides()[3]]);
  return result;
}

void equal_bits(const mx::array& a, const mx::array& b, const std::string& label) {
  require(a.shape() == b.shape(), label + " shape mismatch");
  const auto aa = bits(a), bb = bits(b);
  const auto mismatch = std::mismatch(aa.begin(), aa.end(), bb.begin(), bb.end());
  require(mismatch.first == aa.end(), label + " BF16 bit mismatch at element " +
          std::to_string(std::distance(aa.begin(), mismatch.first)));
}

void verify(const State& state, Pair& oracle) {
  // These short-lived views MUST be destroyed before the next append graph.
  auto view = logical_view(state.backing, state.rows);
  ready(view);
  equal_bits(view.k, oracle.k, "K logical versus concat");
  equal_bits(view.v, oracle.v, "V logical versus concat");
  require(native(view.k).metal_buffer == native(state.backing.k).metal_buffer,
          "logical K view unexpectedly materialized");
  require(native(view.v).metal_buffer == native(state.backing.v).metal_buffer,
          "logical V view unexpectedly materialized");
  if (check_sdpa) {
    auto query = pattern(state.rows, 1, 211);
    auto lhs = mx::fast::scaled_dot_product_attention(query, view.k, view.v,
        1.0f / 16.0f, "", {}, {}, true, gpu());
    auto rhs = mx::fast::scaled_dot_product_attention(query, oracle.k, oracle.v,
        1.0f / 16.0f, "", {}, {}, true, gpu());
    mx::eval(lhs, rhs); lhs.wait(); rhs.wait();
    equal_bits(lhs, rhs, "SDPA logical view versus compact concat");
  }
}

int capacity_for(int required, int limit) {
  require(required > 0 && required <= limit && limit <= 12000, "invalid admitted row bound");
  return std::min(((required + quantum - 1) / quantum) * quantum, limit);
}

void update(Pair& destination, const Pair& source, int first, int end) {
  // Move-assign the new graph before eval; no local old array survives here.
  destination.k = mx::slice_update(destination.k, source.k,
      {0, 0, first, 0}, {1, heads, end, dims}, gpu());
  destination.v = mx::slice_update(destination.v, source.v,
      {0, 0, first, 0}, {1, heads, end, dims}, gpu());
}

Pair zero_capacity(int capacity) {
  Pair value{mx::zeros({1, heads, capacity, dims}, mx::bfloat16, gpu()),
             mx::zeros({1, heads, capacity, dims}, mx::bfloat16, gpu())};
  ready(value);
  return value;
}

State initialize(Pair& oracle, int rows, int limit) {
  State state{zero_capacity(capacity_for(rows, limit)), rows, capacity_for(rows, limit), limit};
  update(state.backing, oracle, 0, rows);
  ready(state.backing);
  require(native(state.backing.k).metal_buffer != native(oracle.k).metal_buffer,
          "initial capacity unexpectedly shares compact oracle");
  require(native(state.backing.v).metal_buffer != native(oracle.v).metal_buffer,
          "initial V capacity unexpectedly shares compact oracle");
  std::cout << "{\"event\":\"initial_conversion\",\"rows\":" << rows << ",\"capacity\":" << state.capacity
            << ",\"compact_k\":"; print_native(native(oracle.k));
  std::cout << ",\"capacity_k\":"; print_native(native(state.backing.k));
  std::cout << "}\n" << std::flush;
  verify(state, oracle);
  return state;
}

void grow(State& state, int capacity) {
  auto old_view = logical_view(state.backing, state.rows);
  ready(old_view);
  auto expanded = zero_capacity(capacity);
  update(expanded, old_view, 0, state.rows);
  state.backing = std::move(expanded);
  state.capacity = capacity;
  // old_view is an input required by the copy graph, released on return.
}

void append(State& state, Pair& oracle, const std::string& label, bool retained_alias = false) {
  if (state.rows >= state.limit) throw std::out_of_range("append exceeds admitted row limit");
  auto row = patterns(state.rows, 1);
  oracle.k = mx::concatenate({oracle.k, row.k}, 2, gpu());
  oracle.v = mx::concatenate({oracle.v, row.v}, 2, gpu());
  ready(oracle); // Independent reference graph must not enter the donation graph.
  const Native old_k = native(state.backing.k), old_v = native(state.backing.v);
  const int old_rows = state.rows, old_capacity = state.capacity;
  const bool growing = state.rows + 1 > state.capacity;
  mx::reset_peak_memory();
  const auto active_before = mx::get_active_memory();
  if (growing) {
    grow(state, capacity_for(state.rows + 1, state.limit));
    ready(state.backing); // Explicit growth phase; its full copy is not hidden.
    ++grow_count;
  }
  update(state.backing, row, state.rows, state.rows + 1);
  ready(state.backing);
  ++state.rows; ++append_count;
  const Native new_k = native(state.backing.k), new_v = native(state.backing.v);
  if (!growing && !retained_alias) {
    require(old_k.donatable && old_v.donatable, "unexpected surviving alias before unaliased append");
    require(same_allocation(old_k, new_k) && same_allocation(old_v, new_v), "unaliased append failed allocation reuse");
  } else {
    require(old_k.metal_buffer != new_k.metal_buffer && old_v.metal_buffer != new_v.metal_buffer,
            "growth/COW did not allocate independent backing");
    if (retained_alias && !growing) ++cow_count;
  }
  std::cout << "{\"event\":\"append\",\"label\":" << std::quoted(label)
            << ",\"old_rows\":" << old_rows << ",\"rows\":" << state.rows
            << ",\"old_capacity\":" << old_capacity << ",\"capacity\":" << state.capacity
            << ",\"growth\":" << growing << ",\"retained_alias\":" << retained_alias
            << ",\"active_before\":" << active_before << ",\"active_after\":" << mx::get_active_memory()
            << ",\"peak_including_oracle_and_update\":" << mx::get_peak_memory() << ",\"old_k\":";
  print_native(old_k); std::cout << ",\"new_k\":"; print_native(new_k);
  std::cout << ",\"old_v\":"; print_native(old_v); std::cout << ",\"new_v\":"; print_native(new_v);
  std::cout << "}\n" << std::flush;
  verify(state, oracle);
}

void sequence(int start, int end, int limit, const std::string& label) {
  auto oracle = patterns(0, start);
  auto state = initialize(oracle, start, limit);
  for (int t = start; t < end; ++t) append(state, oracle, label);
}

void alias_case(bool logical_alias) {
  const int start = 2051;
  auto oracle = patterns(0, start);
  auto state = initialize(oracle, start, 2304);
  const auto before = native(state.backing.k);
  {
    Pair alias = logical_alias ? logical_view(state.backing, state.rows) : state.backing;
    ready(alias);
    const auto old_k_bits = bits(alias.k), old_v_bits = bits(alias.v);
    require(!native(state.backing.k).donatable && !native(state.backing.v).donatable,
            "retained alias did not block donation");
    append(state, oracle, logical_alias ? "logical_view_alias" : "full_backing_alias", true);
    require(native(alias.k).metal_buffer == before.metal_buffer, "old alias changed allocation");
    require(bits(alias.k) == old_k_bits && bits(alias.v) == old_v_bits, "old alias bits were mutated");
  }
  append(state, oracle, "after_alias_release");
}

void clamp_and_reset() {
  {
    auto oracle = patterns(0, 255);
    auto state = initialize(oracle, 255, 257);
    append(state, oracle, "clamp_257"); append(state, oracle, "clamp_257");
    require(state.capacity == 257, "growth exceeded admitted clamp");
    const auto before = native(state.backing.k);
    bool refused = false;
    try { append(state, oracle, "must_refuse"); } catch (const std::out_of_range&) { refused = true; }
    require(refused && state.rows == 257 && same_allocation(before, native(state.backing.k)), "bound rejection mutated state");
    verify(state, oracle);
  }
  mx::synchronize(gpu());
  std::cout << "{\"event\":\"reset\",\"active_bytes\":" << mx::get_active_memory()
            << ",\"allocator_cache_bytes\":" << mx::get_cache_memory() << "}\n";
  // Independent fresh state validates reset without relying on address reuse.
  sequence(254, 255, 512, "fresh_after_reset");
}

int main(int argc, char** argv) {
  std::cout << std::boolalpha << std::setprecision(17);
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg(argv[i]);
      if (arg == "--async-eval") async_mode = true;
      else if (arg == "--sdpa") check_sdpa = true;
      else throw std::invalid_argument("options: --async-eval --sdpa");
    }
    std::cout << "{\"event\":\"start\",\"mlx_version\":" << std::quoted(mx::version())
              << ",\"backend\":\"Metal\",\"shape\":[1,2,\"T\",256],\"dtype\":\"BF16\",\"async_eval\":"
              << async_mode << ",\"sdpa_checked\":" << check_sdpa << "}\n" << std::flush;
    sequence(254, 258, 512, "255_256_257_growth");
    sequence(2050, 2053, 2304, "2051_2052");
    sequence(11230, 11233, 11520, "11232");
    alias_case(false); alias_case(true); clamp_and_reset();
    mx::synchronize(gpu());
    std::cout << "{\"event\":\"summary\",\"passed\":true,\"checks\":" << checks
              << ",\"appends\":" << append_count << ",\"growths\":" << grow_count
              << ",\"cow_appends\":" << cow_count << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "capacity probe failed: " << error.what() << '\n';
    std::cout << "{\"event\":\"summary\",\"passed\":false,\"checks\":" << checks << "}\n";
    return 1;
  }
}
