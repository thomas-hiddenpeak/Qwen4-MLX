// Copyright © 2026 ANERunner contributors.
// Primitive invocation, array ownership and encoder patterns follow MLX / MLX C,
// Copyright © 2023-2025 Apple Inc., distributed under the MIT license.
// This isolated experiment reuses the two original child primitives. It copies
// no Metal kernel and makes no changes to the installed pinned MLX library.

#include "moe_down_pair_bridge.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/c/private/vector.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define ANEMLX_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
using namespace mlx::core;
std::atomic<uint64_t> counts[2]{};
thread_local char last_error[2048]{};

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(std::string("MoE down pair: ") + message);
}

void tensor(const array& value, const Shape& shape, Dtype dtype) {
  require(value.shape() == shape && value.dtype() == dtype, "invalid tensor shape or dtype");
}

void recipe(const array& value, const char* name, size_t inputs, Stream stream) {
  require(value.has_primitive(), "recipe must have a primitive");
  require(value.status() == array::Status::unscheduled && !value.is_tracer(),
          "recipe must be unscheduled and not traced");
  require(value.siblings().empty(), "recipe must have exactly one output");
  require(value.inputs().size() == inputs, "unexpected recipe input count");
  require(value.primitive().stream() == stream, "recipe stream must match explicit GPU stream");
  require(std::strcmp(value.primitive().name(), name) == 0, "unexpected recipe primitive");
}

void shapes(const std::vector<array>& inputs) {
  require(inputs.size() == 10, "expected eight routed and two shared inputs");
  tensor(inputs[0], {10, 640}, bfloat16);
  tensor(inputs[1], {512, 2560, 80}, uint32);
  tensor(inputs[2], {512, 2560, 10}, bfloat16);
  tensor(inputs[3], {512, 2560, 10}, bfloat16);
  tensor(inputs[4], {10}, uint32);
  tensor(inputs[5], {10}, bfloat16);
  tensor(inputs[6], {}, int32);
  tensor(inputs[7], {}, int32);
  tensor(inputs[8], {1, 640}, bfloat16);
  tensor(inputs[9], {640, 2560}, bfloat16);
}

void evaluated_layout(const std::vector<array>& inputs) {
  shapes(inputs);
  for (size_t i = 0; i < 9; ++i) {
    require(inputs[i].flags().row_contiguous, "routed and activation inputs must be row-contiguous");
    require(inputs[i].buffer().ptr() != nullptr, "evaluated input has no storage");
  }
  const auto& weight = inputs[9];
  require(weight.buffer().ptr() != nullptr && weight.flags().col_contiguous &&
              weight.strides() == Strides{1, 640},
          "shared weight must be the original BF16 [2560,640] transpose view");
}

class MoEDownPair final : public Primitive {
 public:
  MoEDownPair(Stream stream, std::shared_ptr<Primitive> routed,
             std::shared_ptr<Primitive> shared, bool serial)
      : Primitive(stream), routed_(std::move(routed)), shared_(std::move(shared)), serial_(serial) {}

  const char* name() const override { return "ANEMLXMoEDownPair"; }

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("MoE down pair is GPU-only");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
    evaluated_layout(inputs);
    require(outputs.size() == 2, "invalid sibling output count");
    tensor(outputs[0], {2560}, bfloat16);
    tensor(outputs[1], {1, 2560}, bfloat16);
    require(outputs[0].id() != outputs[1].id(), "pair outputs must be distinct");
    for (const auto& input : inputs)
      require(input.id() != outputs[0].id() && input.id() != outputs[1].id(),
              "pair output must not be an input");

    // Resolve BOTH input dependencies before either child is dispatched.
    // A plain barrier() leaves the hazard sets intact, which can add a second
    // whole-encoder barrier in front of the shared child. Binding slot zero is
    // harmless here: each original child rebinds every argument before launch.
    auto& encoder = metal::get_command_encoder(stream());
    for (const auto& input : inputs) encoder.set_input_array(input, 0);
    encoder.maybeInsertBarrier();

    std::vector<array> routed_inputs(inputs.begin(), inputs.begin() + 8);
    std::vector<array> shared_inputs(inputs.begin() + 8, inputs.end());
    std::vector<array> routed_outputs{outputs[0]}, shared_outputs{outputs[1]};
    // The fixed CustomKernel has no fill/copy and Matmul uses the original
    // copy-free scalar GEMV. Virtual invocation is the same interface used by
    // mlx/backend/metal/eval.cpp; no concrete hidden eval_gpu symbol is linked.
    routed_->eval_gpu(routed_inputs, routed_outputs);
    if (serial_) encoder.barrier();
    shared_->eval_gpu(shared_inputs, shared_outputs);

    require(outputs[0].buffer().ptr() && outputs[1].buffer().ptr() &&
                outputs[0].buffer().ptr() != outputs[1].buffer().ptr(),
            "children must allocate distinct output buffers");
    for (const auto& input : inputs)
      require(input.buffer().ptr() != outputs[0].buffer().ptr() &&
                  input.buffer().ptr() != outputs[1].buffer().ptr(),
              "children must not donate input buffers");
    // gpu::eval retains the outer inputs/siblings until command completion.
    // The two child allocations are precisely the outer sibling allocations;
    // there are no hidden intermediate arrays or child scheduler invocations.
    counts[serial_ ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
  }

 private:
  std::shared_ptr<Primitive> routed_, shared_;
  bool serial_;
};
} // namespace

ANEMLX_EXPORT int anemlx_moe_down_pair_version(void) { return 1; }
ANEMLX_EXPORT const char* anemlx_moe_down_pair_last_error(void) { return last_error; }
ANEMLX_EXPORT uint64_t anemlx_moe_down_pair_count(int serial_control) {
  return serial_control == 0 || serial_control == 1
      ? counts[serial_control].load(std::memory_order_relaxed) : 0;
}

ANEMLX_EXPORT int anemlx_moe_down_pair(
    mlx_vector_array* results, mlx_array routed_recipe, mlx_array shared_recipe,
    int serial_control, mlx_stream value) {
  last_error[0] = '\0';
  try {
    require(results != nullptr, "null result vector");
    require(serial_control == 0 || serial_control == 1, "serial control must be 0 or 1");
    const auto stream = mlx_stream_get_(value);
    require(stream.device.type == Device::gpu, "explicit stream must use GPU");
    const auto& routed = mlx_array_get_(routed_recipe);
    const auto& shared = mlx_array_get_(shared_recipe);
    recipe(routed, "CustomKernel", 8, stream);
    recipe(shared, "Matmul", 2, stream);
    tensor(routed, {2560}, bfloat16);
    tensor(shared, {1, 2560}, bfloat16);
    std::vector<array> inputs(routed.inputs());
    inputs.insert(inputs.end(), shared.inputs().begin(), shared.inputs().end());
    shapes(inputs);
    for (const auto& input : inputs)
      require(input.id() != routed.id() && input.id() != shared.id(), "recipe depends on the other output");
    auto primitive = std::make_shared<MoEDownPair>(
        stream, routed.primitive_ptr(), shared.primitive_ptr(), serial_control == 1);
    auto outputs = array::make_arrays({{2560}, {1, 2560}}, {bfloat16, bfloat16}, primitive, inputs);
    mlx_vector_array_set_(*results, std::move(outputs));
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    return 1;
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "MoE down pair: unknown C++ exception");
    return 2;
  }
}
