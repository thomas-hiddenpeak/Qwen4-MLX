// Copyright © 2026 ANERunner contributors.
// Array handle and Metal encoder patterns follow MLX / MLX C,
// Copyright © 2023-2025 Apple Inc., distributed under the MIT license.
// This separate experiment links the pinned stock libmlx; it replaces no MLX
// primitive or installed library. See the accompanying build provenance.

#include "moe_gateup_bridge.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

#include <atomic>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ANEMLX_GATEUP_METALLIB
#error "The builder must supply an absolute ANEMLX_GATEUP_METALLIB path"
#endif
#ifndef ANEMLX_GATEUP_LIBRARY_ID
#error "The builder must supply a unique ANEMLX_GATEUP_LIBRARY_ID"
#endif

#define ANEMLX_EXPORT extern "C" __attribute__((visibility("default")))

namespace {
using namespace mlx::core;
constexpr int kHidden = 2560, kIntermediate = 640, kExperts = 512;
std::atomic<uint64_t> dispatches[4]{};
std::atomic<uint64_t> plan_dispatches[2]{}, down_dispatches[2]{}; // BM16, BM32
thread_local char last_error[2048]{};

void record_error(const char* message) noexcept {
  std::snprintf(last_error, sizeof(last_error), "%s", message);
}

void require_tensor(
    const array& value,
    const Shape& shape,
    Dtype dtype,
    const char* name) {
  if (value.shape() != shape || value.dtype() != dtype) {
    throw std::invalid_argument(
        std::string("anemlx_moe_gateup: invalid shape or dtype for ") + name);
  }
}

int bm_slot(int bm) { return bm == 16 ? 0 : bm == 32 ? 1 : -1; }

int plan_rows(int rows, int bm) {
  if (bm_slot(bm) < 0 || rows <= 0 || rows > std::numeric_limits<int>::max() - 31)
    throw std::invalid_argument("anemlx_moe_expert: M must fit positive int32 tile arithmetic and BM must be 16 or 32");
  const int64_t count = (int64_t(rows) + bm - 1) / bm + kExperts;
  if (count > std::numeric_limits<int>::max())
    throw std::invalid_argument("anemlx_moe_expert: plan capacity overflows int32");
  return static_cast<int>(count);
}

Stream require_gpu_stream(mlx_stream value) {
  auto stream = mlx_stream_get_(value);
  if (stream.device.type != Device::gpu)
    throw std::invalid_argument("anemlx_moe_expert: explicit stream must use GPU");
  return stream;
}

void require_contiguous(const std::vector<array>& inputs) {
  for (size_t i = 0; i < inputs.size(); ++i)
    if (!inputs[i].flags().row_contiguous)
      throw std::invalid_argument("anemlx_moe_expert: input " + std::to_string(i) +
                                  " must evaluate row-contiguous");
}

void require_group_size(MTL::ComputePipelineState* kernel, MTL::Size group) {
  if (group.width * group.height * group.depth > kernel->maxTotalThreadsPerThreadgroup())
    throw std::runtime_error("anemlx_moe_expert: unsupported threadgroup size");
}

int activation_rows(const array& input, int hidden) {
  if (input.ndim() != 3 || input.shape(0) <= 1 ||
      input.shape(0) > std::numeric_limits<int>::max() - 31 ||
      input.shape(1) != 1 || input.shape(2) != hidden || input.dtype() != bfloat16)
    throw std::invalid_argument("anemlx_moe_expert: invalid BF16 activation shape");
  return input.shape(0);
}

template <typename Builder>
int checked_result(mlx_array* result, Builder&& builder) noexcept {
  last_error[0] = '\0';
  try {
    if (!result) throw std::invalid_argument("anemlx_moe_expert: null result");
    array value = builder();
    mlx_array_set_(*result, std::move(value));
    return 0;
  } catch (const std::exception& error) {
    record_error(error.what());
    return 1;
  } catch (...) {
    record_error("anemlx_moe_expert: unknown C++ exception");
    return 2;
  }
}

class MoEGateUp final : public UnaryPrimitive {
 public:
  MoEGateUp(Stream stream, int variant)
      : UnaryPrimitive(stream), variant_(variant) {}

  const char* name() const override { return "ANEMLXMoEGateUp"; }

  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("anemlx_moe_gateup: CPU evaluation is unsupported");
  }

  void eval_gpu(const std::vector<array>& inputs, array& output) override {
    // These are the actual evaluated strides, not speculative lazy metadata.
    // Sorted take outputs and the source Q4 banks are contiguous in the runner.
    // Reject unsupported views rather than silently reading the wrong layout
    // or adding nine normalization graph nodes to every fused invocation.
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].flags().row_contiguous) {
        throw std::invalid_argument(
            "anemlx_moe_gateup: input " + std::to_string(i) +
            " must evaluate row-contiguous");
      }
    }

    const int M = output.shape(0), N = kIntermediate, K = kHidden;
    const int bm = 32, bn = variant_ == 0 ? 64 : 32;
    const int wm = 2, wn = variant_ == 0 ? 2 : 1;
    const bool align_m = M % bm == 0;
    const bool align_n = N % bn == 0;
    const bool align_k = K % 64 == 0;
    const std::string kernel_name = variant_ == 0
        ? "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn64_bk64_wm2_wn2"
        : "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1";
    const std::string specialized_name =
        kernel_name + (align_m ? "_am1" : "_am0") + "_an1_ak1";
    const metal::MTLFCList constants{
        {&align_m, MTL::DataTypeBool, 200},
        {&align_n, MTL::DataTypeBool, 201},
        {&align_k, MTL::DataTypeBool, 202}};
    auto& device = metal::device(stream().device);
    auto* library = device.get_library(
        ANEMLX_GATEUP_LIBRARY_ID, ANEMLX_GATEUP_METALLIB);
    auto* kernel = device.get_kernel(
        kernel_name, library, specialized_name, constants);
    const MTL::Size group(32, wn, wm);
    if (group.width * group.height * group.depth >
        kernel->maxTotalThreadsPerThreadgroup()) {
      throw std::runtime_error(
          "anemlx_moe_gateup: pipeline cannot support requested threadgroup");
    }

    output.set_data(allocator::malloc(output.nbytes()));
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    for (int i = 0; i < 9; ++i) encoder.set_input_array(inputs[i], i);
    encoder.set_output_array(output, 9);
    encoder.set_bytes(M, 10);
    encoder.set_bytes(N, 11);
    encoder.set_bytes(K, 12);
    encoder.dispatch_threadgroups(
        MTL::Size((N + bn - 1) / bn, (M + bm - 1) / bm, 1), group);
    // Successful encoding only: not GPU completion, duration, or memory bytes.
    dispatches[variant_].fetch_add(1, std::memory_order_relaxed);
  }

 private:
  int variant_;
};

class MoEExpertPlan final : public UnaryPrimitive {
 public:
  MoEExpertPlan(Stream stream, int bm) : UnaryPrimitive(stream), bm_(bm) {}
  const char* name() const override { return "ANEMLXMoEExpertPlan"; }
  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("anemlx_moe_expert_plan: CPU evaluation is unsupported");
  }
  void eval_gpu(const std::vector<array>& inputs, array& output) override {
    require_contiguous(inputs);
    auto& device = metal::device(stream().device);
    auto* library = device.get_library(ANEMLX_GATEUP_LIBRARY_ID, ANEMLX_GATEUP_METALLIB);
    auto* kernel = device.get_kernel("anemlx_moe_expert_plan_bm" + std::to_string(bm_), library);
    const MTL::Size group(512, 1, 1);
    require_group_size(kernel, group);
    output.set_data(allocator::malloc(output.nbytes()));
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(inputs[0], 0);
    encoder.set_output_array(output, 1);
    const int M = inputs[0].shape(0);
    encoder.set_bytes(M, 2);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), group);
    plan_dispatches[bm_slot(bm_)].fetch_add(1, std::memory_order_relaxed);
  }
 private:
  int bm_;
};

class MoEGroupedProjection final : public UnaryPrimitive {
 public:
  MoEGroupedProjection(Stream stream, int bm, bool gateup)
      : UnaryPrimitive(stream), bm_(bm), gateup_(gateup) {}
  const char* name() const override {
    return gateup_ ? "ANEMLXMoEGateUpPlanned" : "ANEMLXMoEGroupedDown";
  }
  void eval_cpu(const std::vector<array>&, array&) override {
    throw std::runtime_error("anemlx_moe_grouped: CPU evaluation is unsupported");
  }
  void eval_gpu(const std::vector<array>& inputs, array& output) override {
    require_contiguous(inputs);
    const int M = output.shape(0);
    const int N = gateup_ ? kIntermediate : kHidden;
    const int K = gateup_ ? kHidden : kIntermediate;
    const int wm = bm_ == 32 ? 2 : 1;
    const std::string kernel_name =
        std::string(gateup_ ? "anemlx_moe_gateup_grouped" : "anemlx_moe_down_grouped") +
        "_bf16_q4_g64_bm" + std::to_string(bm_) + "_bn32_bk64_wm" + std::to_string(wm) + "_wn1";
    auto& device = metal::device(stream().device);
    auto* library = device.get_library(ANEMLX_GATEUP_LIBRARY_ID, ANEMLX_GATEUP_METALLIB);
    auto* kernel = device.get_kernel(kernel_name, library);
    const MTL::Size group(32, 1, wm);
    require_group_size(kernel, group);
    output.set_data(allocator::malloc(output.nbytes()));
    auto& encoder = metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    const int input_count = gateup_ ? 9 : 5;
    for (int i = 0; i < input_count; ++i) encoder.set_input_array(inputs[i], i);
    encoder.set_output_array(output, input_count);
    encoder.set_bytes(M, input_count + 1);
    encoder.set_bytes(N, input_count + 2);
    encoder.set_bytes(K, input_count + 3);
    // Fixed capacity, never read validTiles back to the CPU. Kernels use the
    // header to return uniformly for unused descriptor threadgroups.
    const int capacity = plan_rows(M, bm_) - 1;
    encoder.dispatch_threadgroups(MTL::Size((N + 31) / 32, capacity, 1), group);
    if (gateup_)
      dispatches[bm_ == 32 ? 2 : 3].fetch_add(1, std::memory_order_relaxed);
    else
      down_dispatches[bm_slot(bm_)].fetch_add(1, std::memory_order_relaxed);
  }
 private:
  int bm_;
  bool gateup_;
};
} // namespace

ANEMLX_EXPORT int anemlx_moe_gateup_version(void) { return 2; }

ANEMLX_EXPORT const char* anemlx_moe_gateup_last_error(void) {
  return last_error;
}

ANEMLX_EXPORT uint64_t anemlx_moe_gateup_dispatch_count(int variant) {
  return variant < 0 || variant > 3 ? 0 :
      dispatches[variant].load(std::memory_order_relaxed);
}

ANEMLX_EXPORT uint64_t anemlx_moe_expert_plan_dispatch_count(int bm) {
  return bm_slot(bm) < 0 ? 0 : plan_dispatches[bm_slot(bm)].load(std::memory_order_relaxed);
}

ANEMLX_EXPORT uint64_t anemlx_moe_grouped_down_dispatch_count(int bm) {
  return bm_slot(bm) < 0 ? 0 : down_dispatches[bm_slot(bm)].load(std::memory_order_relaxed);
}

ANEMLX_EXPORT int anemlx_moe_gateup(
    mlx_array* result,
    mlx_array x,
    mlx_array gate_w,
    mlx_array gate_scale,
    mlx_array gate_bias,
    mlx_array up_w,
    mlx_array up_scale,
    mlx_array up_bias,
    mlx_array indices,
    mlx_array sigmoid_lut,
    int variant) {
  last_error[0] = '\0';
  try {
    if (!result) throw std::invalid_argument("anemlx_moe_gateup: null result");
    if (variant < 0 || variant > 1)
      throw std::invalid_argument("anemlx_moe_gateup: variant must be 0 or 1");
    const auto& input = mlx_array_get_(x);
    if (input.ndim() != 3 || input.shape(0) <= 1 ||
        input.shape(0) > std::numeric_limits<int>::max() - 31 ||
        input.shape(1) != 1 || input.shape(2) != kHidden ||
        input.dtype() != bfloat16) {
      throw std::invalid_argument(
          "anemlx_moe_gateup: expected BF16 x [M>1,1,2560]");
    }
    const int rows = input.shape(0);
    std::vector<array> inputs{
        input, mlx_array_get_(gate_w), mlx_array_get_(gate_scale),
        mlx_array_get_(gate_bias), mlx_array_get_(up_w),
        mlx_array_get_(up_scale), mlx_array_get_(up_bias),
        mlx_array_get_(indices), mlx_array_get_(sigmoid_lut)};
    for (int i : {1, 4})
      require_tensor(inputs[i], {kExperts, kIntermediate, kHidden / 8}, uint32,
                     i == 1 ? "gate_w" : "up_w");
    for (int i : {2, 3, 5, 6})
      require_tensor(inputs[i], {kExperts, kIntermediate, kHidden / 64}, bfloat16,
                     "scale/bias");
    require_tensor(inputs[7], {rows}, uint32, "indices");
    require_tensor(inputs[8], {65536}, bfloat16, "sigmoid_lut");

    // Uses the same thread-default GPU stream as mlx_default_gpu_stream_new().
    // Construct the complete lazy node before touching the destination handle.
    auto stream = default_stream(Device(Device::gpu));
    array value({rows, 1, kIntermediate}, bfloat16,
                std::make_shared<MoEGateUp>(stream, variant), std::move(inputs));
    // Official MLXC setter updates an existing array or allocates when ctx=null.
    // It does not free the caller's handle twice, and retains each input node.
    mlx_array_set_(*result, std::move(value));
    return 0;
  } catch (const std::exception& error) {
    record_error(error.what());
    return 1;
  } catch (...) {
    record_error("anemlx_moe_gateup: unknown C++ exception");
    return 2;
  }
}

ANEMLX_EXPORT int anemlx_moe_expert_plan(
    mlx_array* result, mlx_array indices, int bm, mlx_stream stream) {
  return checked_result(result, [&] {
    const auto& ids = mlx_array_get_(indices);
    if (ids.ndim() != 1 || ids.dtype() != uint32)
      throw std::invalid_argument("anemlx_moe_expert_plan: expected U32 indices[M]");
    const int capacity_with_header = plan_rows(ids.shape(0), bm);
    return array({capacity_with_header, 4}, int32,
                 std::make_shared<MoEExpertPlan>(require_gpu_stream(stream), bm), {ids});
  });
}

ANEMLX_EXPORT int anemlx_moe_gateup_planned(
    mlx_array* result,
    mlx_array x,
    mlx_array gate_w,
    mlx_array gate_scale,
    mlx_array gate_bias,
    mlx_array up_w,
    mlx_array up_scale,
    mlx_array up_bias,
    mlx_array indices,
    mlx_array sigmoid_lut,
    mlx_array plan,
    int variant,
    mlx_stream stream) {
  return checked_result(result, [&] {
    if (variant != 2 && variant != 3)
      throw std::invalid_argument("anemlx_moe_gateup_planned: variant must be 2 or 3");
    const int bm = variant == 2 ? 32 : 16;
    const auto& input = mlx_array_get_(x);
    const int rows = activation_rows(input, kHidden);
    std::vector<array> inputs{
        input, mlx_array_get_(gate_w), mlx_array_get_(gate_scale),
        mlx_array_get_(gate_bias), mlx_array_get_(up_w), mlx_array_get_(up_scale),
        mlx_array_get_(up_bias), mlx_array_get_(plan), mlx_array_get_(sigmoid_lut)};
    for (int i : {1, 4})
      require_tensor(inputs[i], {kExperts, kIntermediate, kHidden / 8}, uint32, "gate/up weight");
    for (int i : {2, 3, 5, 6})
      require_tensor(inputs[i], {kExperts, kIntermediate, kHidden / 64}, bfloat16, "gate/up scale/bias");
    require_tensor(mlx_array_get_(indices), {rows}, uint32, "indices");
    require_tensor(inputs[7], {plan_rows(rows, bm), 4}, int32, "shared expert plan");
    require_tensor(inputs[8], {65536}, bfloat16, "sigmoid LUT");
    return array({rows, 1, kIntermediate}, bfloat16,
                 std::make_shared<MoEGroupedProjection>(require_gpu_stream(stream), bm, true),
                 std::move(inputs));
  });
}

ANEMLX_EXPORT int anemlx_moe_grouped_down(
    mlx_array* result,
    mlx_array activation,
    mlx_array down_w,
    mlx_array down_scale,
    mlx_array down_bias,
    mlx_array plan,
    int bm,
    mlx_stream stream) {
  return checked_result(result, [&] {
    const auto& input = mlx_array_get_(activation);
    const int rows = activation_rows(input, kIntermediate);
    std::vector<array> inputs{input, mlx_array_get_(down_w), mlx_array_get_(down_scale),
                              mlx_array_get_(down_bias), mlx_array_get_(plan)};
    require_tensor(inputs[1], {kExperts, kHidden, kIntermediate / 8}, uint32, "down weight");
    for (int i : {2, 3})
      require_tensor(inputs[i], {kExperts, kHidden, kIntermediate / 64}, bfloat16, "down scale/bias");
    require_tensor(inputs[4], {plan_rows(rows, bm), 4}, int32, "shared expert plan");
    return array({rows, 1, kHidden}, bfloat16,
                 std::make_shared<MoEGroupedProjection>(require_gpu_stream(stream), bm, false),
                 std::move(inputs));
  });
}
