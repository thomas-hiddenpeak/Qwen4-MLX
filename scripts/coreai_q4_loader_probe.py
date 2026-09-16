#!/usr/bin/env python3
"""Isolated Q4 loader experiment: contiguous words and one affine load/thread.

Based on the thread mapping in Apple's MLX QuantizedBlockLoader, inspected in
mlx/backend/metal/kernels/quantized_nax.h. This changes only who loads each
threadgroup weight element; FP32 dequant, FP16 store, barriers and MPP reduction
remain the existing flat grouped implementation. Production defaults unchanged.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch

from coreai_q4_grouped import GEMM_SOURCE, get_plan_kernel, make_plan, grouped_reference
from coreai_q4_flat import _grouped_source, _reshape, get_flat_grouped_kernel


def contiguous_affine_loader(prefix=""):
    """Reusable weight-load block; prefix may be '', 'gate_' or 'up_'."""
    if prefix not in ("", "gate_", "up_"):
        raise ValueError("Unsupported projection prefix")
    return r"""
  {
    constexpr int WORDS_PER_THREAD = BN*(BK/4)/128;
    static_assert(WORDS_PER_THREAD > 0 && (BK/4)%WORDS_PER_THREAD == 0);
    static_assert(16%WORDS_PER_THREAD == 0); // Never cross one group64.
    const int first=int(thread_id)*WORDS_PER_THREAD;
    const int row=first/(BK/4), first_word=first%(BK/4), n=col+row;
    const int first_k=base+first_word*4;
    float scale=0.0f,bias=0.0f;
    if(n<N && first_k<K) {
      scale=float(PREFIXscales[first_k/64,n,expert]);
      bias=float(PREFIXbiases[first_k/64,n,expert]);
    }
    #pragma clang loop unroll(full)
    for(int j=0;j<WORDS_PER_THREAD;++j) {
      const int word=first_word+j,k=base+word*4;
      const ushort bits=(n<N && k<K) ? ushort(PREFIXpacked[k/4,n,expert]) : ushort(0);
      #pragma clang loop unroll(full)
      for(int nibble=0;nibble<4;++nibble) {
        const int code=(uint(bits)>>(nibble*4))&15;
        right_memory[row*BK+word*4+nibble]=half(scale*float(code)+bias);
      }
    }
  }
""".replace("PREFIX", prefix)


def candidate_source(experts, outputs, inputs, block=16, columns=32, inner=64):
    start = GEMM_SOURCE.index("  // Decode each original I16 word once")
    end = GEMM_SOURCE.index("  threadgroup_barrier", start)
    source = GEMM_SOURCE[:start] + contiguous_affine_loader() + GEMM_SOURCE[end:]
    # _grouped_source replaces packed[k/4,...], but affine coordinates were
    # hoisted to first_k. Normalize that spelling before flattening addressing.
    for name in ("scales", "biases"):
        source = source.replace(f"{name}[first_k/64,n,expert]",
            f"{name}[(expert*N+n)*(K/64)+first_k/64]")
    source = source.replace("const int K=int(x.get_extent(0)), N=int(packed.get_extent(1));",
                            f"const int K={inputs}, N={outputs};")
    source = source.replace("packed[k/4,n,expert]", "packed[(expert*N+n)*(K/4)+k/4]")
    return source.replace("BM", str(block)).replace("BN", str(columns)).replace("BK", str(inner))


@cache
def get_loader_kernel(experts, outputs, inputs, block=16, columns=32, inner=64):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if block not in (16, 32) or columns not in (32, 64) or inner not in (64, 128) or inputs % 64:
        raise ValueError("Unsupported bounded tile")

    def reference(x: torch.Tensor, plan: torch.Tensor, packed: torch.Tensor,
                  scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        return grouped_reference(x, plan, *_reshape(packed, scales, biases, experts, outputs, inputs))

    return TorchMetalKernel(f"qwen_q4_contiguous_loader_e{experts}_n{outputs}_k{inputs}_m{block}_n{columns}_k{inner}_v1",
        input_names=["x", "plan", "packed", "scales", "biases"], result_names=["output"],
        src=candidate_source(experts, outputs, inputs, block, columns, inner), torch_defn=reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("thread_id", "uint", "thread_index_in_threadgroup")])


class Projection(torch.nn.Module):
    def __init__(self, experts, outputs, inputs, candidate):
        super().__init__()
        self.geometry = experts, outputs, inputs
        self.candidate = candidate

    def forward(self, x, ids, packed, scales, biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        kernel = (get_loader_kernel if self.candidate else get_flat_grouped_kernel)(*self.geometry, 16, 32, 64)
        result = kernel(x, plan, packed, scales, biases,
            threads_per_grid=(((outputs + 31)//32)*128, plan.shape[0]-1, 1),
            threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return result, plan


def verify_loader_mapping():
    """Check ownership, bounds and affine-group reuse separately from GEMM."""
    cases = []
    for bn in (32, 64):
        for bk in (64, 128):
            words = bn*(bk//4)//128
            visits = []
            for thread in range(128):
                first = thread*words
                row, begin = divmod(first, bk//4)
                assert begin//16 == (begin+words-1)//16
                visits.extend((row, word) for word in range(begin, begin+words))
            assert len(visits) == len(set(visits)) == bn*bk//4
            assert set(visits) == {(row, word) for row in range(bn) for word in range(bk//4)}
            cases.append({"BN": bn, "BK": bk, "wordsPerThread": words,
                          "oldAffinePairs": bn*bk//4, "newAffinePairs": 128})
    return cases


def export_pair(output, examples, geometry):
    import coreai_torch
    output.mkdir(parents=True, exist_ok=False)
    experts, outputs, inputs = geometry
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = [get_plan_kernel(experts, 16), get_flat_grouped_kernel(*geometry, 16, 32, 64),
               get_loader_kernel(*geometry, 16, 32, 64)]
    converter.register_custom_kernels(kernels)
    for candidate, name in ((False, "baseline"), (True, "candidate")):
        module = Projection(*geometry, candidate).eval()
        converter.add_pytorch_module(module, entrypoint_name=name,
            input_names=("x", "ids", "packed", "scales", "biases"), output_names=("output", "plan"),
            export_fn=lambda m: torch.export.export(m, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output/"projection.aimodel")
    (output/"candidate.metalbody").write_text(candidate_source(*geometry))


def tiny(output):
    from coreai_q4_metal import make_smoke
    from export_coreai_q4_moe import tensor_json
    original, x, ids = make_smoke(79, 256, 67, 5)
    ids, order = torch.sort(ids)
    x = x[:, 0][order]
    args = (x, ids, original.packed.flatten(), original.scales.flatten(), original.biases.flatten())
    with torch.inference_mode():
        baseline = Projection(5, 67, 256, False)(*args)
        candidate = Projection(5, 67, 256, True)(*args)
    assert all(torch.equal(a, b) for a, b in zip(baseline, candidate))
    export_pair(output, args, (5, 67, 256))
    fixture = {"inputs": {name: tensor_json(value) for name, value in
                         zip(("x", "ids", "packed", "scales", "biases"), args)},
               "expectedOutputs": {name: tensor_json(value) for name, value in zip(("output", "plan"), baseline)}}
    (output/"actual.json").write_text(json.dumps(fixture)+"\n")
    (output/"cpu-check.json").write_text(json.dumps({"tinyCPUCallbackExact": True,
        "loaderMapping": verify_loader_mapping(), "deviceValidated": False}, indent=2)+"\n")


def real_shape(output, manifest_path, rows):
    manifest = json.loads(manifest_path.read_text())
    layer = next(layer for layer in manifest["layers"] if layer["index"] == 0)
    weights = layer["weights"]
    records = {record["bufferName"].rsplit(".", 1)[-1]: record for record in weights["buffers"]
               if record["bufferName"].startswith("moe.decode.down_proj.")}
    geometry = (512, 2560, 640)
    assert set(records) == {"packed", "scales", "biases"}
    weight_file = (manifest_path.parent/weights["path"]).resolve()
    args = (torch.empty(rows, 640, dtype=torch.float16), torch.empty(rows, dtype=torch.int32),
            *(torch.empty(record["byteLength"]//2, dtype=torch.int16 if name == "packed" else torch.float16)
              for name in ("packed", "scales", "biases") for record in [records[name]]))
    # Only shape propagation/opaque fake callbacks run; real weight bytes stay
    # in their existing file and are never loaded or expanded by this author.
    export_pair(output, args, geometry)
    generator = np.random.default_rng(927641)
    with (output/"x.bin").open("wb") as f:
        for start in range(0, rows, 1024):
            value = (generator.standard_normal((min(1024, rows-start), 640), dtype=np.float32)*0.25).astype(np.float16)
            f.write(value.tobytes())
    ids = (np.arange(rows, dtype=np.int64)*512//rows).astype(np.int32)
    ids.tofile(output/"ids.bin")
    inputs = {"x": {"file": "x.bin", "offset": 0, "bytes": rows*640*2, "shape": [rows, 640], "dtype": "float16"},
              "ids": {"file": "ids.bin", "offset": 0, "bytes": rows*4, "shape": [rows], "dtype": "int32"}}
    for name, record in records.items():
        inputs[name] = {"file": str(weight_file), "offset": record["byteOffset"],
                        "bytes": record["byteLength"], "shape": [record["byteLength"]//2], "dtype": record["dtype"]}
    for name in ("baseline", "candidate"):
        spec = {"asset": "projection.aimodel", "function": name, "inputs": inputs,
                "output": name+"-output", "repeats": 12, "mapped": False, "oneBufferPerFile": False}
        (output/(name+"-spec.json")).write_text(json.dumps(spec, indent=2)+"\n")
    (output/"manifest.json").write_text(json.dumps({"status": "CPU-authored-device-unvalidated", "rows": rows,
        "geometry": {"E": 512, "N": 2560, "K": 640}, "tile": [16, 32, 64],
        "fixture": "Seed927641 random FP16 activations, uniformly sorted expert assignments, real layer0 down weights",
        "ownedInputPolicy": "Separate MTLBuffer per input; same bytes and files for both entrypoints",
        "loaderMapping": verify_loader_mapping()}, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rows", type=int, default=20480)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.manifest:
        if args.rows <= 0 or args.rows > 81920:
            raise ValueError("Use positive rows <=81920")
        real_shape(args.output, args.manifest, args.rows)
    else:
        tiny(args.output)
