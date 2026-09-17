#!/usr/bin/env python3
"""CPU-author a tiny CoreAI tensor-reference/device-pointer ABI probe."""
import argparse
import json
from pathlib import Path

import torch


SOURCE = r'''
const device half* xp=&x[0,0];
const device short* pp=&packed[0];
const device uchar* bytes=reinterpret_cast<const device uchar*>(pp);
device half* yp=&output[0,0];
const int i=int(index),columns=int(x.get_extent(0));
if(i<columns*int(x.get_extent(1))) {
  const int row=i/columns,column=i%columns;
  yp[row*int(output.get_stride(1))+column]=xp[row*int(x.get_stride(1))+column];
}
if(i<8)unpacked[i]=int(bytes[i]);
'''


def main():
    import coreai_torch
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    from export_coreai_q4_moe import tensor_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)

    def reference(x: torch.Tensor, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x.clone(), packed.view(torch.uint8).int()

    kernel = TorchMetalKernel('qwen_nax_tensor_pointer_abi_v1',
        input_names=['x', 'packed'], result_names=['output', 'unpacked'], src=SOURCE,
        torch_defn=reference,
        metal_params=[MetalParameter('index', 'uint', 'thread_position_in_grid')])

    class Probe(torch.nn.Module):
        def forward(self, x, packed):
            return kernel(x, packed, threads_per_grid=(32, 1, 1),
                threads_per_thread_group=(32, 1, 1), result_shapes=[list(x.shape), [8]])

    x = (torch.arange(15).reshape(3, 5)-7).half()/8
    packed = torch.tensor([-32768, -1, 0x1234, 32767], dtype=torch.int16)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([kernel])
    converter.add_pytorch_module(Probe(), input_names=('x', 'packed'), output_names=('output', 'unpacked'),
        export_fn=lambda module: torch.export.export(module, args=(x, packed)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(args.output/'abi.aimodel')
    expected = reference(x, packed)
    (args.output/'actual.json').write_text(json.dumps({'inputs': {'x': tensor_json(x), 'packed': tensor_json(packed)},
        'expectedOutputs': dict(zip(('output', 'unpacked'), map(tensor_json, expected), strict=True))})+'\n')
    inputs = {}
    for name, value in (('x', x), ('packed', packed)):
        value.numpy().tofile(args.output/(name+'.bin'))
        inputs[name] = {'file': name+'.bin', 'offset': 0, 'bytes': value.numel()*2,
            'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    (args.output/'spec.json').write_text(json.dumps({'asset': 'abi.aimodel', 'function': 'main',
        'inputs': inputs, 'output': 'device-output', 'repeats': 2, 'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    for name, source in kernel.kernel_cache.values():
        (args.output/(name+'.metal')).write_text(source)


if __name__ == '__main__':
    main()
