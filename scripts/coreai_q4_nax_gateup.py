#!/usr/bin/env python3
"""Optional NAX fused grouped gate/up; one shared weight tile, reused A registers.

Uses the same MIT-attributed fragment mapping as coreai_q4_nax, and preserves the
existing fused gate/up rounding policy. No default production kernel is changed.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch

from coreai_q4_nax import MLX_NOTICE, check_fragment_mapping
from coreai_q4_flat import _reshape, get_flat_gateup_kernel
from coreai_q4_gateup import gateup_reference
from coreai_q4_grouped import get_plan_kernel, make_plan


def _weight_load(prefix):
    return r'''
  {
    const int n=col+int(thread_id);
    float scale=0.0f,bias=0.0f;
    if(n<N) {
      const int affine=(expert*N+n)*(K/64)+base/64;
      scale=float(PREFIXsp[affine]);bias=float(PREFIXbp[affine]);
    }
    #pragma clang loop unroll(full)
    for(int word=0;word<16;++word) {
      const int packed_index=(expert*N+n)*(K/4)+base/4+word;
      const ushort bits=n<N ? ushort(PREFIXwp[packed_index]) : ushort(0);
      #pragma clang loop unroll(full)
      for(int nibble=0;nibble<4;++nibble) {
        const int code=(uint(bits)>>(nibble*4))&15;
        right_memory[int(thread_id)*72+word*4+nibble]=half(scale*float(code)+bias);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  #pragma clang loop unroll(full)
  for(int part=0;part<4;++part) {
    #pragma clang loop unroll(full)
    for(int i=0;i<8;++i) {
      const int row=frag_row+(i/4)*8,k=part*16+frag_col+i%4;
      a[i]=cached_a[part*8+i];
      b[i]=right_memory[(simd*32+row)*72+k];
      b[8+i]=right_memory[(simd*32+16+row)*72+k];
    }
    operation.run(a,b,PREFIXaccum);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
'''.replace('PREFIX', prefix)


def source_text(outputs, inputs, simdgroups=1):
    if inputs % 64 or min(outputs, inputs) < 1 or simdgroups not in (1, 2, 4):
        raise ValueError('Expected positive group64 geometry and1/2/4SIMD groups')
    source = r'''
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1],start=plan[1,tile+1],count=plan[2,tile+1];
constexpr int K=INPUTS,N=OUTPUTS,BN=32*SIMDGROUPS;
const int lane=int(thread_id)%32,simd=int(thread_id)/32,col=int(group.x)*BN;
const device half* xp=&x[0,0];
const device short* gate_wp=&gate_packed[0];
const device half* gate_sp=&gate_scales[0];
const device half* gate_bp=&gate_biases[0];
const device short* up_wp=&up_packed[0];
const device half* up_sp=&up_scales[0];
const device half* up_bp=&up_biases[0];
device half* yp=&output[0,0];
const int x_stride=int(x.get_stride(1)),y_stride=int(output.get_stride(1));
const int qid=lane>>2,frag_row=(qid&4)|((lane>>1)&3),frag_col=((qid&2)|(lane&1))*4;
threadgroup half right_memory[BN*72];
constexpr auto desc=matmul2d_descriptor(16,32,16,false,true,true,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroup> operation;
auto a=operation.get_left_input_cooperative_tensor<half,half,float>();
auto b=operation.get_right_input_cooperative_tensor<half,half,float>();
auto gate_accum=operation.get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
auto up_accum=operation.get_destination_cooperative_tensor<
    metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();
#pragma clang loop unroll(full)
for(int i=0;i<16;++i) {gate_accum[i]=0.0f;up_accum[i]=0.0f;}
for(int base=0;base<K;base+=64) {
  // Four A fragments stay private to each lane and feed both projections.
  // Only one BK72 weight tile exists; gate/up reuse it between barriers.
  thread half cached_a[32];
  #pragma clang loop unroll(full)
  for(int part=0;part<4;++part) {
    #pragma clang loop unroll(full)
    for(int i=0;i<8;++i) {
      const int row=frag_row+(i/4)*8,k=base+part*16+frag_col+i%4;
      cached_a[part*8+i]=row<count ? xp[(start+row)*x_stride+k] : half(0);
    }
  }
GATE_BODY
UP_BODY
}
#pragma clang loop unroll(full)
for(int i=0;i<16;++i) {
  const int row=frag_row+((i%8)/4)*8;
  const int n=col+simd*32+(i/8)*16+frag_col+i%4;
  if(row<count && n<N) {
    const half gate=half(gate_accum[i]),up=half(up_accum[i]);
    const half sigmoid=half(1.0f/(1.0f+exp(-float(gate))));
    const half silu=half(float(gate)*float(sigmoid));
    yp[(start+row)*y_stride+n]=half(float(silu)*float(up));
  }
}
'''
    return source.replace('GATE_BODY', _weight_load('gate_')).replace('UP_BODY', _weight_load('up_')).replace(
        'INPUTS', str(inputs)).replace('OUTPUTS', str(outputs)).replace('SIMDGROUPS', str(simdgroups))


def get_kernel(experts, outputs, inputs, simdgroups=1):
    return _get_kernel(experts, outputs, inputs, simdgroups)


@cache
def _get_kernel(experts, outputs, inputs, simdgroups):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor, up_packed: torch.Tensor,
                  up_scales: torch.Tensor, up_biases: torch.Tensor) -> torch.Tensor:
        return gateup_reference(x, plan,
            *_reshape(gate_packed, gate_scales, gate_biases, experts, outputs, inputs),
            *_reshape(up_packed, up_scales, up_biases, experts, outputs, inputs))

    return TorchMetalKernel(f'qwen_q4_nax_gateup_e{experts}_n{outputs}_k{inputs}_sg{simdgroups}_ptr_v1',
        input_names=['x', 'plan', 'gate_packed', 'gate_scales', 'gate_biases', 'up_packed', 'up_scales', 'up_biases'],
        result_names=['output'], src=source_text(outputs, inputs, simdgroups), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Probe(torch.nn.Module):
    def __init__(self, geometry, candidate, simdgroups=1):
        super().__init__()
        self.geometry, self.candidate, self.simdgroups = geometry, candidate, simdgroups

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        if self.candidate:
            kernel = get_kernel(*self.geometry, self.simdgroups)
            columns, threads = self.simdgroups*32, self.simdgroups*32
        else:
            kernel = get_flat_gateup_kernel(*self.geometry, 16, 32, 64)
            columns, threads = 32, 128
        output = kernel(x, plan, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            threads_per_grid=(((outputs+columns-1)//columns)*threads, plan.shape[0]-1, 1),
            threads_per_thread_group=(threads, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


NAMES = ('x', 'ids', 'gate_packed', 'gate_scales', 'gate_biases', 'up_packed', 'up_scales', 'up_biases')


def export_pair(path, examples, geometry, simdgroups=1):
    import coreai_torch
    path.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernel = get_kernel(*geometry, simdgroups)
    converter.register_custom_kernels([get_plan_kernel(geometry[0], 16),
        get_flat_gateup_kernel(*geometry, 16, 32, 64), kernel])
    for candidate, name in ((False, 'baseline'), (True, 'candidate')):
        converter.add_pytorch_module(Probe(geometry, candidate, simdgroups).eval(),
            entrypoint_name=name, input_names=NAMES, output_names=('output', 'plan'),
            export_fn=lambda module: torch.export.export(module, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path/'gateup.aimodel')
    for name, source in kernel.kernel_cache.values():
        (path/(name+'.metal')).write_text(source)


def main():
    from coreai_q4_metal import make_smoke
    from export_coreai_q4_moe import tensor_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, help='Prior gateup pair directory; reuse exact inputs')
    parser.add_argument('--simdgroups', type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check_fragment_mapping()
    if args.fixture:
        previous = args.fixture.resolve()
        inputs = json.loads((previous/'baseline-spec.json').read_text())['inputs']
        geometry = (512, 640, 2560)
        assert inputs['x']['shape'][1] == 2560
        assert all(inputs[name]['shape'] == [512*640*2560//4] for name in ('gate_packed', 'up_packed'))
        examples = tuple(torch.empty(inputs[name]['shape'], dtype=torch.int32 if name == 'ids' else
            torch.int16 if name.endswith('packed') else torch.float16) for name in NAMES)
        export_pair(args.output, examples, geometry, args.simdgroups)
        for value in inputs.values():
            value['file'] = str((previous/value['file']).resolve())
    else:
        gate, x, ids = make_smoke(79, 128, 67, 5)
        up, _, _ = make_smoke(79, 128, 67, 5)
        up.packed.bitwise_xor_(0x1234)
        up.scales.mul_(.5)
        ids, order = torch.sort(ids)
        examples = (x[:, 0][order], ids, gate.packed.flatten(), gate.scales.flatten(), gate.biases.flatten(),
                    up.packed.flatten(), up.scales.flatten(), up.biases.flatten())
        geometry = (5, 67, 128)
        with torch.inference_mode():
            expected = Probe(geometry, False)(*examples)
            candidate = Probe(geometry, True, args.simdgroups)(*examples)
        assert all(torch.equal(a, b) for a, b in zip(expected, candidate, strict=True))
        export_pair(args.output, examples, geometry, args.simdgroups)
        (args.output/'actual.json').write_text(json.dumps({'inputs': dict(zip(NAMES, map(tensor_json, examples), strict=True)),
            'expectedOutputs': dict(zip(('output', 'plan'), map(tensor_json, expected), strict=True))})+'\n')
        inputs = {}
        for name, value in zip(NAMES, examples, strict=True):
            value.numpy().tofile(args.output/(name+'.bin'))
            inputs[name] = {'file': name+'.bin', 'offset': 0, 'bytes': value.numel()*value.element_size(),
                'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    for name in ('baseline', 'candidate'):
        (args.output/(name+'-spec.json')).write_text(json.dumps({'asset': 'gateup.aimodel', 'function': name,
            'inputs': inputs, 'output': name+'-output', 'repeats': 12, 'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'geometry': dict(zip(('E', 'N', 'K'), geometry)), 'BM': 16, 'BN': 32*args.simdgroups,
        'BK': 64, 'simdgroups': args.simdgroups, 'weightSharedStride': 72, 'weightSharedTiles': 1,
        'activationReuse': 'Four private A fragments loaded once perBK64 for both gate and up',
        'outputPolicy': 'Both completed accumulators→half, half sigmoid, half gate*sigmoid, half *up',
        'fixtureSource': str(args.fixture) if args.fixture else 'tiny CPU fixture',
        'MLXRuntimeLinked': False, 'license': 'Apple MLX MIT notice embedded in emitted Metal'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
