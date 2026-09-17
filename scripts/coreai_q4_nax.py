#!/usr/bin/env python3
"""Isolated CoreAI grouped down projection with MLX-style NAX registers.

The fragment coordinates, per-SIMD MPP operand layout, and BK+8 half weight
padding are adapted from Apple's MIT-licensed MLX quantized_nax.h and
steel/gemm/nax.h. Only the required 16x32x16 register operation is retained;
there is no MLX library import/link or runtime ownership. Leaf IO uses the
existing CoreAI tensor ABI, with an optional separately probed pointer path.

Source: https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch

from coreai_q4_flat import _reshape, _validate, get_flat_grouped_kernel
from coreai_q4_grouped import get_plan_kernel, grouped_reference, make_plan


MLX_NOTICE = r'''/*
Portions adapted from MLX steel/gemm/nax.h (Copyright © 2025 Apple Inc.)
and quantized_nax.h (Copyright © 2023-2024 Apple Inc.).
https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels

MIT License
Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/'''


BODY = r'''
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1],start=plan[1,tile+1],count=plan[2,tile+1];
constexpr int K=INPUTS,N=OUTPUTS,BN=32*SIMDGROUPS,THREADS=32*SIMDGROUPS;
const int lane=int(thread_id)%32,simd=int(thread_id)/32;
const int col=int(group.x)*BN;
POINTER_DECLARATIONS

// MLX BaseNAXFrag::get_coord, expressed through the public thread attribute.
// Each lane owns two rows separated by8, and four adjacent columns per row.
const int qid=lane>>2;
const int frag_row=(qid&4)|((lane>>1)&3);
const int frag_col=((qid&2)|(lane&1))*4;
threadgroup half right_memory[BN*72];

// MLX register MMA: A16x16, B32x16, C16x32, transposeC=true.
constexpr auto desc=matmul2d_descriptor(16,32,16,false,true,true,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> operation;
auto a=operation.get_left_input_cooperative_tensor<half,half,float>();
auto b=operation.get_right_input_cooperative_tensor<half,half,float>();
auto c=operation.get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
#pragma clang loop unroll(full)
for(int i=0;i<16;++i)c[i]=0.0f;

for(int base=0;base<K;base+=64) {
  // Exactly one adjacent group64 output row per thread. Weight affine math
  // remains FP32 scale*code+bias -> FP16, identical to the existing kernel.
  const int n=col+int(thread_id);
  float scale=0.0f,bias=0.0f;
  if(n<N) {
    const int affine=(expert*N+n)*(K/64)+base/64;
    scale=float(READ_SCALE);
    bias=float(READ_BIAS);
  }
  #pragma clang loop unroll(full)
  for(int word=0;word<16;++word) {
    const int packed_index=(expert*N+n)*(K/4)+base/4+word;
    const ushort bits=n<N ? ushort(READ_PACKED) : ushort(0);
    #pragma clang loop unroll(full)
    for(int nibble=0;nibble<4;++nibble) {
      const int code=(uint(bits)>>(nibble*4))&15;
      right_memory[int(thread_id)*72+word*4+nibble]=half(scale*float(code)+bias);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Direct device→register activations; only dequantized weights use shared
  // memory. No left tile or threadgroup-wide MMA is introduced.
  #pragma clang loop unroll(disable)
  for(int kk=0;kk<64;kk+=16) {
    #pragma clang loop unroll(full)
    for(int i=0;i<8;++i) {
      const int row=frag_row+(i/4)*8,k=base+kk+frag_col+i%4;
      a[i]=row<count ? READ_X : half(0);
      b[i]=right_memory[(simd*32+row)*72+kk+frag_col+i%4];
      b[8+i]=right_memory[(simd*32+16+row)*72+kk+frag_col+i%4];
    }
    operation.run(a,b,c);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
#pragma clang loop unroll(full)
for(int i=0;i<8;++i) {
  const int row=frag_row+(i/4)*8;
  const int n=col+simd*32+frag_col+i%4;
  if(row<count && n<N)STORE_LO;
  if(row<count && n+16<N)STORE_HI;
}
'''


def source_text(outputs, inputs, simdgroups=1, pointers=False):
    if inputs % 64 or min(outputs, inputs) < 1 or simdgroups not in (1, 2, 4):
        raise ValueError('Expected positive group64 shape and1/2/4SIMD groups')
    replacements = {'INPUTS': str(inputs), 'OUTPUTS': str(outputs), 'SIMDGROUPS': str(simdgroups)}
    if pointers:
        replacements.update(POINTER_DECLARATIONS='''const device half* xp=&x[0,0];
const device short* wp=&packed[0];
const device half* sp=&scales[0];
const device half* bp=&biases[0];
device half* yp=&output[0,0];
const int x_stride=int(x.get_stride(1)),y_stride=int(output.get_stride(1));''',
            READ_SCALE='sp[affine]', READ_BIAS='bp[affine]', READ_PACKED='wp[packed_index]',
            READ_X='xp[(start+row)*x_stride+k]',
            STORE_LO='yp[(start+row)*y_stride+n]=half(c[i])',
            STORE_HI='yp[(start+row)*y_stride+n+16]=half(c[8+i])')
    else:
        replacements.update(POINTER_DECLARATIONS='', READ_SCALE='scales[affine]',
            READ_BIAS='biases[affine]', READ_PACKED='packed[packed_index]',
            READ_X='x[k,start+row]', STORE_LO='output[n,start+row]=half(c[i])',
            STORE_HI='output[n+16,start+row]=half(c[8+i])')
    source = BODY
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def get_kernel(experts, outputs, inputs, simdgroups=1, pointers=False):
    return _get_kernel(experts, outputs, inputs, simdgroups, pointers)


@cache
def _get_kernel(experts, outputs, inputs, simdgroups, pointers):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, packed: torch.Tensor,
                  scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        return grouped_reference(x, plan, *_reshape(packed, scales, biases, experts, outputs, inputs))

    return TorchMetalKernel(f'qwen_q4_nax_down_e{experts}_n{outputs}_k{inputs}_sg{simdgroups}'+('_ptr' if pointers else '_tensor')+'_v1',
        input_names=['x', 'plan', 'packed', 'scales', 'biases'], result_names=['output'],
        src=source_text(outputs, inputs, simdgroups, pointers), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


def nax_grouped_linear(x, plan, projection, simdgroups=1):
    _validate(x, projection.packed, projection.scales, projection.biases, *projection.geometry)
    columns = threads = 32*simdgroups
    return get_kernel(*projection.geometry, simdgroups, True)(x, plan,
        projection.packed, projection.scales, projection.biases,
        threads_per_grid=(((projection.output_size+columns-1)//columns)*threads, plan.shape[0]-1, 1),
        threads_per_thread_group=(threads, 1, 1), result_shapes=[[x.shape[0], projection.output_size]])


def nax_grouped_gateup(x, plan, gate, up, simdgroups=1):
    from coreai_q4_nax_gateup import get_kernel as get_gateup_kernel
    if gate.geometry != up.geometry:
        raise ValueError('NAX gate/up geometry must match')
    for projection in (gate, up):
        _validate(x, projection.packed, projection.scales, projection.biases, *projection.geometry)
    columns = threads = 32*simdgroups
    return get_gateup_kernel(*gate.geometry, simdgroups)(x, plan,
        gate.packed, gate.scales, gate.biases, up.packed, up.scales, up.biases,
        threads_per_grid=(((gate.output_size+columns-1)//columns)*threads, plan.shape[0]-1, 1),
        threads_per_thread_group=(threads, 1, 1), result_shapes=[[x.shape[0], gate.output_size]])


def nax_moe_kernels(moe):
    from coreai_q4_nax_gateup import get_kernel as get_gateup_kernel
    fused_gateup = moe.fuse_gateup and 'gate_proj' in moe.nax_projections
    names = tuple(name for name in moe.nax_projections
                  if not (fused_gateup and name in ('gate_proj', 'up_proj')))
    kernels = [get_kernel(*getattr(moe.decode, name).geometry, moe.nax_simdgroups, True) for name in names]
    if fused_gateup:
        kernels.append(get_gateup_kernel(*moe.decode.gate_proj.geometry, moe.nax_simdgroups))
    return list(dict.fromkeys(kernels))


def install_nax_moe(module, *, projections='down', simdgroups=1):
    """Opt in only routed prefill projections; no tensor storage or S1 changes.

    The default selects only GPU-validated bitwise-exact down projections.
    projections='all' also selects the experimental gate/up implementation.
    Install after flatten_moe_weights. Requires the existing BM16 plan contract;
    NAX supersedes other flat grouped-loader choices. The returned kernels are
    additions to existing registrations; custom_kernels() also includes them.
    """
    from coreai_moe_chunk import ChunkQ4MoE
    chunks = [child for child in module.modules() if isinstance(child, ChunkQ4MoE)]
    if not chunks or any(not moe.flat_weights for moe in chunks):
        raise ValueError('NAX installation requires flattened ChunkQ4MoE weights')
    if simdgroups not in (1, 2, 4):
        raise ValueError('NAX requires1/2/4SIMD groups')
    if projections not in ('down', 'all'):
        raise ValueError('NAX projections must be down or all')
    if any((moe.block, moe.columns, moe.inner) != (16, 32, 64) for moe in chunks):
        raise ValueError('NAX installation currently requires existing16/32/64tile and BM16 plan')
    before = [(name, value.dtype, tuple(value.shape), value.data_ptr()) for name, value in module.named_buffers()]
    kernels = []
    for moe in chunks:
        moe.nax_simdgroups = simdgroups
        moe.nax_projections = ('down_proj',) if projections == 'down' else ('gate_proj', 'up_proj', 'down_proj')
        kernels += nax_moe_kernels(moe)
    for moe in chunks:
        moe.nax_moe = True
    after = [(name, value.dtype, tuple(value.shape), value.data_ptr()) for name, value in module.named_buffers()]
    if before != after:
        raise AssertionError('NAX installation changed learned buffer metadata or storage')
    return list(dict.fromkeys(kernels))


class Projection(torch.nn.Module):
    def __init__(self, geometry, candidate, simdgroups=1, pointers=False):
        super().__init__()
        self.geometry, self.candidate = geometry, candidate
        self.simdgroups, self.pointers = simdgroups, pointers

    def forward(self, x, ids, packed, scales, biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        if self.candidate:
            kernel = get_kernel(*self.geometry, self.simdgroups, self.pointers)
            threads, columns = self.simdgroups*32, self.simdgroups*32
        else:
            kernel = get_flat_grouped_kernel(*self.geometry, 16, 32, 64)
            threads, columns = 128, 32
        output = kernel(x, plan, packed, scales, biases,
            threads_per_grid=(((outputs+columns-1)//columns)*threads, plan.shape[0]-1, 1),
            threads_per_thread_group=(threads, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def export_pair(path, examples, geometry, simdgroups=1, pointers=False):
    import coreai_torch
    path.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernel = get_kernel(*geometry, simdgroups, pointers)
    converter.register_custom_kernels([get_plan_kernel(geometry[0], 16),
        get_flat_grouped_kernel(*geometry, 16, 32, 64), kernel])
    for candidate, name in ((False, 'baseline'), (True, 'candidate')):
        converter.add_pytorch_module(Projection(geometry, candidate, simdgroups, pointers).eval(),
            entrypoint_name=name, input_names=('x', 'ids', 'packed', 'scales', 'biases'), output_names=('output', 'plan'),
            export_fn=lambda module: torch.export.export(module, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path/'down.aimodel')
    for name, source in kernel.kernel_cache.values():
        (path/(name+'.metal')).write_text(source)


def check_fragment_mapping():
    coordinates = []
    for lane in range(32):
        qid = lane>>2
        row, col = (qid&4)|((lane>>1)&3), ((qid&2)|(lane&1))*4
        coordinates.extend((row+(i//4)*8, col+i%4) for i in range(8))
    assert len(set(coordinates)) == len(coordinates) == 256
    assert set(coordinates) == {(row, col) for row in range(16) for col in range(16)}


def main():
    from export_coreai_q4_moe import tensor_json
    from coreai_q4_metal import make_smoke
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, help='Prior down pair directory; reuse exact raw input files')
    parser.add_argument('--simdgroups', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--pointers', action='store_true', help='Requires successful tiny pointer ABI device probe')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check_fragment_mapping()
    if args.fixture:
        previous = args.fixture.resolve()
        spec = json.loads((previous/'baseline-spec.json').read_text())
        inputs = spec['inputs']
        geometry = (512, 2560, 640)
        assert inputs['x']['shape'][1] == 640
        assert inputs['packed']['shape'] == [512*2560*640//4]
        examples = tuple(torch.empty(inputs[name]['shape'], dtype=torch.int32 if name == 'ids' else torch.int16 if name == 'packed' else torch.float16)
                         for name in ('x', 'ids', 'packed', 'scales', 'biases'))
        export_pair(args.output, examples, geometry, args.simdgroups, args.pointers)
        for entry in inputs.values():
            entry['file'] = str((previous/entry['file']).resolve())
    else:
        projection, x, ids = make_smoke(79, 128, 67, 5)
        ids, order = torch.sort(ids)
        examples = (x[:, 0][order], ids, projection.packed.flatten(), projection.scales.flatten(), projection.biases.flatten())
        geometry = (5, 67, 128)
        with torch.inference_mode():
            baseline = Projection(geometry, False)(*examples)
            candidate = Projection(geometry, True, args.simdgroups, args.pointers)(*examples)
        assert all(torch.equal(a, b) for a, b in zip(baseline, candidate, strict=True))
        export_pair(args.output, examples, geometry, args.simdgroups, args.pointers)
        (args.output/'actual.json').write_text(json.dumps({
            'inputs': dict(zip(('x', 'ids', 'packed', 'scales', 'biases'), map(tensor_json, examples), strict=True)),
            'expectedOutputs': dict(zip(('output', 'plan'), map(tensor_json, baseline), strict=True))})+'\n')
        inputs = {}
        for name, value in zip(('x', 'ids', 'packed', 'scales', 'biases'), examples, strict=True):
            value.numpy().tofile(args.output/(name+'.bin'))
            inputs[name] = {'file': name+'.bin', 'offset': 0, 'bytes': value.numel()*value.element_size(),
                'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    for name in ('baseline', 'candidate'):
        (args.output/(name+'-spec.json')).write_text(json.dumps({'asset': 'down.aimodel', 'function': name,
            'inputs': inputs, 'output': name+'-output', 'repeats': 12, 'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'source': 'Apple MLX quantized_nax.h and steel/gemm/nax.h; MIT notice embedded in emitted Metal',
        'geometry': dict(zip(('E', 'N', 'K'), geometry)), 'BM': 16, 'BN': 32*args.simdgroups,
        'BK': 64, 'weightSharedStride': 72, 'activationSharedBytes': 0,
        'simdgroups': args.simdgroups, 'MPP': 'Per-SIMD register16x32x16 FP16 operands/FP32 accumulator',
        'pointers': args.pointers, 'fixtureSource': str(args.fixture) if args.fixture else 'tiny deterministic mixed experts',
        'CPUCallbackExact': True if not args.fixture else None, 'fragmentCoordinateCoverageExact': True,
        'MLXRuntimeLinked': False, 'accumulation': 'K16 register operations; requires device numerical comparison to original K64 operation'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
