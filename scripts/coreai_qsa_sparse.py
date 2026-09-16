#!/usr/bin/env python3
"""Experimental direct selected-block QSA attention, CPU authoring only.

One SIMDgroup owns one (query token, KV head), covering up to 16 GQA heads.
MPP QK/PV operate on 16x32x32 tiles. Only 32 selected key positions and a
16x32 probability tile are staged; there is no global per-query K/V gather.
Softmax statistics and output accumulators are FP32. Probabilities round to
FP16 before PV, and the final output rounds to FP16. This candidate does not
modify the validated QSA wrapper or any runtime/exporter default.

Primary API: local macOS27 MPPTensorOpsMatMul2d.h and
https://developer.apple.com/videos/play/wwdc2026/330/
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_qsa_chunk import write_tensor_json
from export_coreai_qsa import compare
from export_moe import sha256_file, write_json


INPUTS = ("query", "keys", "values", "block_ids", "offset")


def selected_positions(block_ids, position, capacity, ratio=4, *, preserve_slots=False):
    """Bounded eager oracle preserving selected-score order, then live tail.

    IDs must be unique among visible complete blocks. Out-of-range, negative,
    and future IDs are ignored. Early queries attend every live position:
    top-k necessarily contains every visible full block at that boundary.
    """
    if not 0 <= position < capacity:
        raise ValueError("Query position outside cache")
    ids = block_ids.tolist()
    complete = (position + 1) // ratio
    if complete <= len(ids):
        return list(range(position + 1))
    visible = [int(block) for block in ids if 0 <= int(block) < complete and (int(block) + 1) * ratio <= capacity]
    if len(visible) != len(set(visible)):
        raise ValueError("Selected visible block IDs must be unique")
    if preserve_slots:
        slots = [int(block) * ratio + lane if 0 <= int(block) < complete and (int(block) + 1) * ratio <= capacity else -1
                 for block in ids for lane in range(ratio)]
    else:
        slots = [block * ratio + lane for block in visible for lane in range(ratio)]
    return slots + list(range(complete * ratio, position + 1))


def sparse_reference(query, keys, values, block_ids, offset, *, ratio=4):
    """FP32 semantic oracle, gathering one query at a time on CPU only."""
    if isinstance(query, FakeTensor) or query.device.type == "meta":
        return torch.empty_like(query)
    count, width, heads = query.shape[2], query.shape[3], query.shape[1]
    kv_heads, capacity = keys.shape[1:3]
    repeats = heads // kv_heads
    out = torch.zeros_like(query)
    start = int(offset.item())
    for token in range(count):
        positions = selected_positions(block_ids[0, token], start + token, capacity, ratio)
        if not positions:
            continue
        for kv in range(kv_heads):
            q = query[0, kv * repeats:(kv + 1) * repeats, token].float()
            k, v = keys[0, kv, positions].float(), values[0, kv, positions].float()
            probability = torch.softmax((q @ k.T) * (width ** -0.5), dim=-1)
            out[0, kv * repeats:(kv + 1) * repeats, token] = (probability @ v).half()
    return out


def online_reference(query, keys, values, block_ids, offset, *, ratio=4):
    """CPU algorithm oracle for tile32 online softmax and half-probability PV.

    MPP reduction order remains device-dependent; this verifies the intended
    recurrence and mixed-precision boundaries, not the uncompiled Metal body.
    """
    count, width, heads = query.shape[2], query.shape[3], query.shape[1]
    kv_heads, capacity = keys.shape[1:3]
    repeats = heads // kv_heads
    result = torch.zeros_like(query)
    for token in range(count):
        positions = selected_positions(block_ids[0, token], int(offset.item()) + token, capacity, ratio, preserve_slots=True)
        for kv in range(kv_heads):
            q = query[0, kv * repeats:(kv + 1) * repeats, token].float()
            maximum = torch.full((repeats, 1), -torch.inf)
            denominator = torch.zeros(repeats, 1)
            accumulator = torch.zeros(repeats, width)
            for start in range(0, len(positions), 32):
                chunk = positions[start:start + 32]
                valid = torch.tensor([position >= 0 for position in chunk])
                gather = [max(position, 0) for position in chunk]
                k = torch.where(valid[:, None], keys[0, kv, gather].float(), 0)
                v = torch.where(valid[:, None], values[0, kv, gather].float(), 0)
                scores = torch.zeros(repeats, len(chunk))
                for inner in range(0, width, 32):
                    scores += q[:, inner:inner + 32] @ k[:, inner:inner + 32].T
                scores *= width ** -0.5
                scores = torch.where(valid[None], scores, -torch.inf)
                next_maximum = torch.maximum(maximum, scores.max(-1, keepdim=True).values)
                alpha = torch.where(torch.isfinite(maximum), torch.exp(maximum - next_maximum), 0)
                probability = torch.where(torch.isfinite(scores), torch.exp(scores - next_maximum), 0)
                denominator = denominator * alpha + probability.sum(-1, keepdim=True)
                accumulator = accumulator * alpha + probability.half().float() @ v
                maximum = next_maximum
            output = torch.where(denominator > 0, accumulator / denominator.clamp_min(1e-30), 0)
            result[0, kv * repeats:(kv + 1) * repeats, token] = output.half()
    return result


def select_blocks(scores, offset, *, ratio=4, budget=2048):
    """Existing score bias/visibility/top-k semantics, with unsorted IDs."""
    positions = offset + torch.arange(scores.shape[1], dtype=torch.int32)
    blocks = torch.arange(scores.shape[2], dtype=torch.int32)
    visible = (blocks[None] + 1) * ratio <= positions[:, None] + 1
    biased = scores.float() - blocks.float() * 1e-7
    return torch.where(visible[None], biased, -torch.inf).topk(budget // ratio, -1, sorted=False).indices.to(torch.int32)


METAL_PREFIX = r"""
const int token = int(group.x);
const int kv = int(group.y);
const int count = int(query.get_extent(1));
const int capacity = int(keys.get_extent(1));
const int topk = int(block_ids.get_extent(0));
const int position = offset[0] + token;
if (token >= count || kv >= int(keys.get_extent(2))) return;
const int complete = (position + 1) / RATIO;
const bool dense = complete <= topk;
const int tail_start = complete * RATIO;
const int active = dense ? position + 1 : topk * RATIO + position + 1 - tail_start;
threadgroup int positions[32];
threadgroup half probabilities[16 * 32];
threadgroup float tile_maximum[16];
threadgroup float tile_sum[16];
threadgroup float running_maximum[16];
threadgroup float denominator[16];
threadgroup float alpha[16];
auto max_tensor = tensor<threadgroup float, extents<int,16>, tensor_inline>(tile_maximum, extents<int,16>());
auto sum_tensor = tensor<threadgroup float, extents<int,16>, tensor_inline>(tile_sum, extents<int,16>());
if (lane < 16u) {
    running_maximum[lane] = -INFINITY;
    denominator[lane] = 0.0f;
    alpha[lane] = 0.0f;
}
constexpr auto qk_descriptor = matmul2d_descriptor(16,32,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
constexpr auto pv_descriptor = matmul2d_descriptor(16,32,32,false,false,false,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_descriptor,execution_simdgroup> qk;
matmul2d<pv_descriptor,execution_simdgroup> pv;
auto q_tile = qk.get_left_input_cooperative_tensor<half,half,float>();
auto k_tile = qk.get_right_input_cooperative_tensor<half,half,float>();
auto scores = qk.get_destination_cooperative_tensor<decltype(q_tile),decltype(k_tile),float>();
auto row_max = qk.get_row_reduction_destination_cooperative_tensor<decltype(q_tile),decltype(k_tile),float>();
auto row_sum = qk.get_row_reduction_destination_cooperative_tensor<decltype(q_tile),decltype(k_tile),float>();
auto p_tile = pv.get_left_input_cooperative_tensor<half,half,float>();
auto v_tile = pv.get_right_input_cooperative_tensor<half,half,float>();
ACCUMULATOR_DECLARATIONS
threadgroup_barrier(mem_flags::mem_threadgroup);
for (int base = 0; base < active; base += 32) {
    const int slot = base + int(lane);
    int key_position = -1;
    if (slot < active) {
        if (dense) {
            key_position = slot;
        } else if (slot < topk * RATIO) {
            const int block = block_ids[slot / RATIO, token, 0];
            if (block >= 0 && block < complete) key_position = block * RATIO + slot % RATIO;
        } else {
            key_position = tail_start + slot - topk * RATIO;
        }
    }
    // Both lower and upper bounds are checked before any cache read.
    positions[lane] = key_position >= 0 && key_position <= position && key_position < capacity ? key_position : -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < scores.get_capacity(); ++i)
        if (scores.is_valid_element(i)) scores[i] = 0.0f;
    for (int inner = 0; inner < HEAD_DIM; inner += 32) {
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < q_tile.get_capacity(); ++i) if (q_tile.is_valid_element(i)) {
            const auto at = q_tile.get_multidimensional_index(i);
            const int d = inner + int(at[0]), row = int(at[1]);
            q_tile[i] = row < GQA ? query[d, token, kv * GQA + row, 0] : half(0);
        }
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < k_tile.get_capacity(); ++i) if (k_tile.is_valid_element(i)) {
            const auto at = k_tile.get_multidimensional_index(i);
            const int d = inner + int(at[0]), column = int(at[1]);
            const int key_index = positions[column];
            k_tile[i] = key_index >= 0 ? keys[d, key_index, kv, 0] : half(0);
        }
        qk.run(q_tile,k_tile,scores);
    }
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < scores.get_capacity(); ++i) if (scores.is_valid_element(i)) {
        const auto at = scores.get_multidimensional_index(i);
        scores[i] = int(at[1]) < GQA && positions[int(at[0])] >= 0 ? scores[i] * SCALE : -INFINITY;
    }
    reduce_rows(scores,row_max,reduction_operation::max,-INFINITY);
    row_max.store(max_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16u) {
        const float next = max(running_maximum[lane], tile_maximum[lane]);
        alpha[lane] = isfinite(running_maximum[lane]) ? exp(running_maximum[lane] - next) : 0.0f;
        running_maximum[lane] = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < scores.get_capacity(); ++i) if (scores.is_valid_element(i)) {
        const auto at = scores.get_multidimensional_index(i);
        const int row = int(at[1]), column = int(at[0]);
        const float probability = isfinite(scores[i]) ? exp(scores[i] - running_maximum[row]) : 0.0f;
        scores[i] = probability;
        probabilities[row * 32 + column] = half(probability);
    }
    reduce_rows(scores,row_sum,reduction_operation::sum,0.0f);
    row_sum.store(sum_tensor);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16u) denominator[lane] = denominator[lane] * alpha[lane] + tile_sum[lane];
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < p_tile.get_capacity(); ++i) if (p_tile.is_valid_element(i)) {
        const auto at = p_tile.get_multidimensional_index(i);
        p_tile[i] = probabilities[int(at[1]) * 32 + int(at[0])];
    }
    PV_UPDATES
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
OUTPUT_STORES
"""


def metal_source(head_dim, gqa, ratio):
    declarations, updates, stores = [], [], []
    for tile in range(head_dim // 32):
        declarations.append(f"""auto output_{tile} = pv.get_destination_cooperative_tensor<decltype(p_tile),decltype(v_tile),float>();
    #pragma clang loop unroll(full)
    for (uint16_t i=0;i<output_{tile}.get_capacity();++i) if(output_{tile}.is_valid_element(i)) output_{tile}[i]=0.0f;""")
        updates.append(f"""
    #pragma clang loop unroll(full)
    for (uint16_t i=0;i<v_tile.get_capacity();++i) if(v_tile.is_valid_element(i)) {{
        const auto at=v_tile.get_multidimensional_index(i);
        const int d={tile * 32}+int(at[0]), key_index=positions[int(at[1])];
        v_tile[i]=key_index>=0 ? values[d,key_index,kv,0] : half(0);
    }}
    #pragma clang loop unroll(full)
    for (uint16_t i=0;i<output_{tile}.get_capacity();++i) if(output_{tile}.is_valid_element(i)) {{
        const auto at=output_{tile}.get_multidimensional_index(i);
        output_{tile}[i] *= alpha[int(at[1])];
    }}
    pv.run(p_tile,v_tile,output_{tile});""")
        stores.append(f"""
#pragma clang loop unroll(full)
for (uint16_t i=0;i<output_{tile}.get_capacity();++i) if(output_{tile}.is_valid_element(i)) {{
    const auto at=output_{tile}.get_multidimensional_index(i);
    const int d={tile * 32}+int(at[0]), row=int(at[1]);
    if(row<GQA) output[d,token,kv*GQA+row,0]=denominator[row]>0.0f ? half(output_{tile}[i]/denominator[row]) : half(0);
}}""")
    body = METAL_PREFIX.replace("ACCUMULATOR_DECLARATIONS", "\n".join(declarations))
    body = body.replace("PV_UPDATES", "\n".join(updates)).replace("OUTPUT_STORES", "\n".join(stores))
    return body.replace("HEAD_DIM", str(head_dim)).replace("GQA", str(gqa)).replace("RATIO", str(ratio)).replace("SCALE", f"{head_dim ** -0.5:.12g}f")


@cache
def get_sparse_kernel(head_dim=256, gqa=12, ratio=4):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if head_dim not in (32, 64, 128, 256) or not 1 <= gqa <= 16 or ratio != 4:
        raise ValueError("Bounded prototype requires D32/64/128/256, 1..16 GQA heads, and ratio4")

    def reference(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                  block_ids: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return sparse_reference(query, keys, values, block_ids, offset, ratio=ratio)

    return TorchMetalKernel(f"qwen_qsa_sparse_d{head_dim}_g{gqa}_r{ratio}_tile32_v1",
        input_names=list(INPUTS), result_names=["output"], src=metal_source(head_dim, gqa, ratio), torch_defn=reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("lane", "uint", "thread_index_in_simdgroup")])


class SparseAttention(torch.nn.Module):
    def __init__(self, ratio=4):
        super().__init__()
        self.ratio = ratio

    def forward(self, query, keys, values, block_ids, offset):
        if (query.ndim != 4 or keys.ndim != 4 or values.shape != keys.shape or query.shape[0] != 1 or
            keys.shape[0] != 1 or min(query.shape[1:]) <= 0 or min(keys.shape[1:]) <= 0 or
            query.shape[-1] != keys.shape[-1] or query.shape[1] % keys.shape[1] or
            block_ids.shape[:2] != (1, query.shape[2]) or block_ids.ndim != 3 or offset.shape != (1,) or
            block_ids.shape[-1] <= 0 or keys.shape[2] % self.ratio or
            any(x.dtype != torch.float16 for x in (query, keys, values)) or
            block_ids.dtype != torch.int32 or offset.dtype != torch.int32):
            raise ValueError("Expected Q/K/V half GQA, block_ids[1,S,K] I32, offset[1] I32")
        return get_sparse_kernel(query.shape[-1], query.shape[1] // keys.shape[1], self.ratio)(
            query, keys, values, block_ids, offset,
            threads_per_grid=(query.shape[2] * 32, keys.shape[1], 1), threads_per_thread_group=(32, 1, 1),
            result_shapes=[list(query.shape)])


def make_cases(*, count=7, capacity=128, head_dim=256, gqa=12, kv_heads=2, budget=32, seed=732):
    if capacity % 4 or budget % 4 or not 4 <= budget < capacity or not 1 <= count < capacity - budget:
        raise ValueError("Expected a capacity/budget multiple of4 and count fitting dense+sparse cases")
    generator = torch.Generator().manual_seed(seed)
    query = (torch.randn(1, gqa * kv_heads, count, head_dim, generator=generator) * 0.3).half()
    keys = (torch.randn(1, kv_heads, capacity, head_dim, generator=generator) * 0.3).half()
    values = (torch.randn(1, kv_heads, capacity, head_dim, generator=generator) * 0.3).half()
    scores = torch.rand(1, count, capacity // 4, generator=generator)
    cases = []
    for name, start in (("dense_cold", 0), ("threshold", budget - 1), ("sparse_partial", capacity - count - 1)):
        offset = torch.tensor([start], dtype=torch.int32)
        ids = select_blocks(scores, offset, budget=budget)
        inputs = dict(zip(INPUTS, (query, keys, values, ids, offset)))
        cases.append((name, inputs))
    sparse = cases[-1][1]
    cases.append(("score_order_permuted", {**sparse, "block_ids": sparse["block_ids"].flip(-1).contiguous()}))
    offset = torch.tensor([budget + 7], dtype=torch.int32)
    ids = select_blocks(scores, offset, budget=budget)
    ids[:, :, 0] = capacity // 4 - 1
    ids[:, :, 1] = -1
    cases.append(("invalid_future_sentinels", dict(zip(INPUTS, (query, keys, values, ids, offset)))))
    return cases


def block_ids_from_mask(mask, offset, *, budget=2048, ratio=4):
    """Recover chronological block IDs, requiring exact original-mask replay."""
    if mask.ndim != 4 or mask.shape[:2] != (1, 1) or mask.shape[-1] % ratio or budget % ratio:
        raise ValueError("Expected binary [1,1,S,C] mask with complete capacity blocks")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Mask must contain only zero/one")
    count, capacity = mask.shape[2:]
    topk = budget // ratio
    if not 0 <= offset or offset + count > capacity:
        raise ValueError("Mask offset/count exceeds capacity")
    ids = torch.full((1, count, topk), -1, dtype=torch.int32)
    reference = mask[0, 0].bool()
    reconstructed = torch.zeros_like(reference)
    for token in range(count):
        position = offset + token
        complete = (position + 1) // ratio
        chosen = torch.where(reference[token].reshape(-1, ratio).all(-1))[0]
        chosen = chosen[chosen < complete]
        if len(chosen) != min(topk, complete):
            raise ValueError(f"Mask query{token} does not have the expected complete-block selection")
        ids[0, token, :len(chosen)] = chosen.to(torch.int32)
        positions = selected_positions(ids[0, token], position, capacity, ratio)
        reconstructed[token, positions] = True
    if not torch.equal(reconstructed, reference):
        raise ValueError("Recovered blocks/tail do not exactly reconstruct the original mask")
    return ids


def _save_probe_asset(output, inputs):
    import coreai_torch
    module = SparseAttention().eval()
    kernel = get_sparse_kernel(inputs["query"].shape[-1], inputs["query"].shape[1] // inputs["keys"].shape[1], 4)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([kernel])
    converter.add_pytorch_module(module, input_names=INPUTS, output_names=("output",),
        export_fn=lambda m: torch.export.export(m, args=tuple(inputs[name] for name in INPUTS)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output / "sparse.aimodel")
    (output / "coreai-main-after.txt").write_text(str(program.get_graph("main")))
    for identifier, source in kernel.kernel_cache.values():
        (output / (identifier + ".metal")).write_text(source)


def export_benchmark(output, *, sdpa_fixture=None, offset=8192, count=2048,
                     capacity=16384, head_dim=256, budget=2048):
    """One exact-replay or synthetic large case; no eager half SDPA call."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if sdpa_fixture is not None:
        fixture_path = Path(sdpa_fixture).resolve()
        data = json.loads(fixture_path.read_text())
        tensors = {}
        # Release each parsed number list immediately after tensor conversion.
        for name, value in data.pop("inputs").items():
            dtype = {"float16": torch.float16, "float32": torch.float32, "int32": torch.int32}[value["dtype"]]
            tensors[name] = torch.tensor(value.pop("values"), dtype=dtype).reshape(value["shape"])
        reference = data["expectedOutputs"]["attention"]
        expected = torch.tensor(reference.pop("values"), dtype=torch.float16).reshape(reference["shape"])
        del data
        query, keys, values = (tensors[name] for name in ("query", "keys", "values"))
        if any(t.dtype != torch.float16 for t in (query, keys, values)):
            raise ValueError("Use the exact half-input SDPA fixture")
        ids = block_ids_from_mask(tensors["mask"], offset, budget=budget)
        del tensors
        inputs = dict(zip(INPUTS, (query, keys, values, ids, torch.tensor([offset], dtype=torch.int32))))
        with torch.inference_mode():
            actual = sparse_reference(*inputs.values())
        comparison = compare(actual, expected)
        if not comparison["finite"] or comparison["max_abs"] > 0.02 or (comparison["relative_l2"] or 0) > 0.005:
            raise AssertionError(f"Recovered exact-mask sparse oracle differs from source SDPA: {comparison}")
        provenance = {"kind": "exact_existing_SDPA_fixture_replay", "sourceFixture": str(fixture_path),
            "sourceFixtureSHA256": sha256_file(fixture_path), "maskReconstructionExact": True,
            "selectionOrder": "ascending block position recovered from original mask; not original top-k score order",
            "cpuComparisonToExistingSDPA": comparison,
            "activationSource": "Inputs and expected output preserved exactly; capture provenance is inherited from the original fixture's authoring report"}
    else:
        if capacity % 4 or budget % 4 or not 4 <= budget < capacity or not 0 <= offset or offset + count > capacity:
            raise ValueError("Invalid synthetic benchmark dimensions")
        generator = torch.Generator().manual_seed(2051732)
        query = (torch.randn(1, 24, count, head_dim, generator=generator) * 0.3).half()
        keys = (torch.randn(1, 2, capacity, head_dim, generator=generator) * 0.3).half()
        values = (torch.randn(1, 2, capacity, head_dim, generator=generator) * 0.3).half()
        scores = torch.rand(1, count, capacity // 4, generator=generator)
        cursor = torch.tensor([offset], dtype=torch.int32)
        ids = select_blocks(scores, cursor, budget=budget)
        del scores
        inputs = dict(zip(INPUTS, (query, keys, values, ids, cursor)))
        print("Computing bounded per-query FP32 semantic reference", flush=True)
        with torch.inference_mode():
            expected = sparse_reference(*inputs.values())
        provenance = {"kind": "synthetic_full_shape", "seed": 2051732,
            "selectionOrder": "unsorted top-k score order; existing visibility and tie-bias applied",
            "reference": "FP32 attention gathering one query at a time on CPU; no global per-query K/V allocation"}
    _save_probe_asset(output, inputs)
    write_tensor_json(output / "fixture.json", {"output": expected}, inputs=inputs)
    report = {"status": "cpu-authored-real-device-validation-pending", "model": "sparse.aimodel", "function": "main",
        "fixture": "fixture.json", "count": query.shape[2], "capacity": keys.shape[2], "headDimension": query.shape[-1],
        "queryHeads": query.shape[1], "kvHeads": keys.shape[1], "budget": budget, "offset": offset,
        "provenance": provenance, "inputNames": INPUTS, "outputNames": ["output"],
        "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005},
        "explicitThreadgroupScratchBytes": 1472, "globalGatherBuffers": 0,
        "precision": "FP16 Q/K/V, FP32 QK/online max/sum/output accumulators, FP16 probability tile before PV, FP16 output",
        "sourceSHA256": sha256_file(Path(__file__)), "fixtureSHA256": sha256_file(output / "fixture.json"),
        "limitations": ["This script does not execute or compile GPU kernels.",
            "Existing small D256 GPU validation does not prove this larger shape's numerics or performance.",
            "CPU reference uses semantic FP32 attention; actual tiled mixed-precision arithmetic must be compared on device."]}
    write_json(output / "manifest.json", report)
    return report


def export_smoke(output, *, count=7, capacity=128, head_dim=256, budget=32):
    import coreai_torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = make_cases(count=count, capacity=capacity, head_dim=head_dim, budget=budget)
    module = SparseAttention().eval()
    kernel = get_sparse_kernel(head_dim, 12, 4)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([kernel])
    converter.add_pytorch_module(module, input_names=INPUTS, output_names=("output",),
        export_fn=lambda m: torch.export.export(m, args=tuple(cases[0][1].values())).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output / "sparse.aimodel")
    (output / "coreai-main-after.txt").write_text(str(program.get_graph("main")))
    reports = []
    for name, inputs in cases:
        with torch.inference_mode():
            expected = sparse_reference(*inputs.values())
            online = online_reference(*inputs.values())
        metric = compare(online, expected)
        if not metric["finite"] or metric["max_abs"] > 0.02 or (metric["relative_l2"] or 0) > 0.005:
            raise AssertionError(f"CPU mixed-precision online reference failed: {name}: {metric}")
        write_tensor_json(output / (name + ".json"), {"output": expected}, inputs=inputs)
        reports.append({"name": name, "fixture": name + ".json", "onlineCPUComparison": metric})
    for identifier, source in kernel.kernel_cache.values():
        (output / (identifier + ".metal")).write_text(source)
    report = {"status": "cpu-authored-metal-uncompiled-device-unvalidated", "model": "sparse.aimodel", "function": "main",
        "count": count, "capacity": capacity, "headDimension": head_dim, "gqaHeadsPerKV": 12, "kvHeads": 2,
        "budget": budget, "cases": reports, "inputNames": INPUTS, "outputNames": ["output"],
        "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005},
        "explicitThreadgroupScratchBytes": 1472, "globalGatherBuffers": 0,
        "precision": "FP16 Q/K/V, FP32 QK/online max/sum/output accumulators, FP16 probability tile before PV, FP16 output",
        "selectedIDsContract": "Unique visible complete blocks in any score order; invalid/future IDs ignored; dense early queries ignore IDs",
        "sourceSHA256": sha256_file(Path(__file__)),
        "limitations": ["No Metal compilation or device execution was performed by this authoring script.",
            "Fixtures are synthetic dimensions/activations, not real layer3 outputs; full model integration remains untouched.",
            "CPU online oracle verifies recurrence and rounding, not compiler/hardware reduction order or performance.",
            "Per-SIMDgroup D256 output accumulators may cause register pressure; measure before adoption.",
            "Caller must ensure offset>=0 and offset+queryCount<=capacity; selected IDs must be unique among visible complete blocks."],
        "primaryAPI": "https://developer.apple.com/videos/play/wwdc2026/330/"}
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=7)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--head-dim", type=int, choices=(32, 64, 128, 256), default=256)
    parser.add_argument("--budget", type=int, default=32)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sdpa-fixture", type=Path, help="Replay an existing half-input SDPA fixture with exact reconstructed mask")
    mode.add_argument("--large-synthetic", action="store_true", help="Author one supplied full-shape synthetic benchmark")
    parser.add_argument("--offset", type=int, default=8192)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.sdpa_fixture is not None or args.large_synthetic:
        report = export_benchmark(args.output, sdpa_fixture=args.sdpa_fixture, offset=args.offset,
                                  count=args.count, capacity=args.capacity, head_dim=args.head_dim, budget=args.budget)
    else:
        report = export_smoke(args.output, count=args.count, capacity=args.capacity, head_dim=args.head_dim, budget=args.budget)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
