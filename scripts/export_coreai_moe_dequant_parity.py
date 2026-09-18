#!/usr/bin/env python3
"""CPU-author a native-copy/NAX versus parity-aware dequant-once MoE pair.

Reuses already extracted real layer0 bytes and a prefix of the previous S8192
synthetic activation file. No model/device execution and no learned-weight copy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from coreai_expert_grouping import enable_integer_grouping
from coreai_moe_chunk import ChunkQ4MoE, compare, make_synthetic
from coreai_moe_dequant_parity import ParityDequantOnceChunkMoE
from coreai_moe_transfers import install_moe_transfers
from coreai_q4_flat import flatten_moe_weights
from coreai_q4_nax import install_nax_moe
from export_coreai_pd_shared import ExternalModule
from export_coreai_q4_moe import tensor_json
from export_moe import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('coreai_moe_dequant_parity', 'coreai_moe_dequant', 'coreai_moe_chunk',
           'coreai_q4_nax', 'coreai_q4_nax_m32_probe', 'coreai_q4_nax_gateup_parity',
           'coreai_moe_transfers', 'coreai_moe_inverse_copy', 'coreai_expert_grouping',
           'export_coreai_moe_dequant_parity')


def seed():
    base = ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=True).eval()
    flatten_moe_weights(base)
    enable_integer_grouping(base)
    install_moe_transfers(base, tail_precision='native-copy')
    return base


def configure(base, threshold):
    install_nax_moe(base, projections='all', down_block=32, gateup_policy='native-parity-v2')
    return ParityDequantOnceChunkMoE(base, minimum_chunk=threshold, block=32, columns=64).eval()


def author(directory, base, candidate, x, named, input_names, specs, *, small):
    import coreai_torch
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    args = (x, *(value for _, value in named))
    kernels = list(dict.fromkeys(base.custom_kernels()+candidate.custom_kernels()))
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    for name, model in (('baseline', base), ('candidate', candidate)):
        wrapper = ExternalModule(model, [n for n, _ in named], 1).eval()
        def export(current, name=name):
            ep = torch.export.export(current, args).run_decompositions(coreai_torch.get_decomp_table())
            placeholders = [n for n in ep.graph.nodes if n.op == 'placeholder']
            captured = [s.target for s, n in zip(ep.graph_signature.input_specs, placeholders, strict=True)
                        if str(s.kind).endswith('BUFFER') and len(n.users)]
            if any(n != 'base.decode.expert_ids' for n in captured):
                raise ValueError(f'Unexpected learned constants: {captured}')
            (directory/f'{name}-torch-graph.txt').write_text(ep.graph_module.code)
            return ep
        converter.add_pytorch_module(wrapper, entrypoint_name=name, input_names=('x', *input_names),
                                     output_names=('output', 'ids', 'scores'), export_fn=export)
    program = converter.to_coreai()
    program.optimize()
    (directory/'optimized.mlir').write_text(str(program._mlir_module))
    asset = directory/'moe.aimodel'
    program.save_asset(asset)
    size = sum(p.stat().st_size for p in asset.rglob('*') if p.is_file())
    if size >= 1_000_000:
        raise ValueError('External pair unexpectedly captured learned weights')
    for kernel in kernels:
        for name, source in kernel.kernel_cache.values():
            (directory/(name+'.metal')).write_text(source)
    for name in ('baseline', 'candidate'):
        write_json(directory/f'{name}-spec.json', {'asset': 'moe.aimodel', 'function': name,
            'inputs': specs, 'output': f'device-{name}', 'repeats': 10,
            'mapped': False, 'oneBufferPerFile': False})
    report = {'assetBytes': size, 'authorSeconds': time.perf_counter()-started,
              'learnedConstantCapture': False, 'deviceValidated': False}
    if small:
        with torch.inference_mode():
            actual, expected = candidate(x), base(x)
        report['cpuComparison'] = dict(zip(('output', 'ids', 'scores'),
                                          (compare(a, b) for a, b in zip(actual, expected, strict=True))))
        write_json(directory/'actual.json', {'inputs': dict(zip(('x', *input_names), map(tensor_json, args))),
                                            'expectedOutputs': dict(zip(('output', 'ids', 'scores'), map(tensor_json, expected)))})
    write_json(directory/'manifest.json', report)
    print('READY '+str(directory), flush=True)
    return report


def small_probe(output):
    base = seed()
    candidate = configure(base, 4)
    x = (torch.randn(1, 17, 64, generator=torch.Generator().manual_seed(997))*.5).half()
    x[:, 0] = 0
    named = [(n, v) for n, v in base.named_buffers() if n != 'decode.expert_ids']
    names = [f'weight_{i:03d}' for i in range(len(named))]
    inputs = output/'small-inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    specs = {}
    for name, value in zip(('x', *names), (x, *(v for _, v in named)), strict=True):
        path = inputs/(name+'.bin')
        value.numpy().tofile(path)
        specs[name] = {'file': str(path.resolve()), 'offset': 0, 'bytes': value.numel()*value.element_size(),
                       'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
    return author(output/'small', base, candidate, x, named, names, specs, small=True)


def real_probe(output, fixture, count):
    previous = json.loads((fixture/'manifest.json').read_text())
    records = previous['sourceWeightRecords']
    source_specs = json.loads((fixture/'real/baseline-spec.json').read_text())['inputs']
    base = seed()
    base.experts = base.decode.experts = 512
    base.hidden = base.decode.hidden = 2560
    base.decode.expert_ids = torch.arange(512, dtype=torch.int32)
    named, names, specs, hashes = [], [], {}, {}
    for record in records:
        name = record['bufferName'].removeprefix('moe.')
        input_name = record['inputName']
        spec = dict(source_specs[input_name])
        path = Path(spec['file'])
        if not path.is_absolute():
            path = (fixture/'real'/path).resolve()
        spec['file'] = str(path)
        if spec['offset'] != 0 or path.stat().st_size != spec['bytes']:
            raise ValueError('Expected prior one-file-per-weight exact bytes')
        hashes[input_name] = sha256_file(path)
        if hashes[input_name] != record['sha256']:
            raise ValueError('Prior extracted weight SHA256 differs from original model record')
        value = torch.from_numpy(np.memmap(path, dtype={'float16': '<f2', 'int16': '<i2'}[spec['dtype']],
                                          mode='c', shape=tuple(spec['shape'])))
        owner = base
        parts = name.split('.')
        for part in parts[:-1]:
            owner = getattr(owner, part)
        setattr(owner, parts[-1], value)
        named.append((name, value))
        names.append(input_name)
        specs[input_name] = spec
    for projection in ('gate_proj', 'up_proj', 'down_proj'):
        bank = getattr(base.decode, projection)
        record = next(r for r in records if r['bufferName'] == f'moe.decode.{projection}.packed')
        bank.expert_count, bank.output_size, words = record['shape']
        bank.input_size = words*4
    assert [n for n, _ in named] == [n for n, _ in base.named_buffers() if n != 'decode.expert_ids']
    candidate = configure(base, count)
    xspec = dict(source_specs['x'])
    xspec['shape'] = [1, count, 2560]
    xspec['bytes'] = count*2560*2
    if not Path(xspec['file']).is_absolute():
        xspec['file'] = str((fixture/'real'/xspec['file']).resolve())
    x = torch.from_numpy(np.memmap(xspec['file'], dtype='<f2', mode='c', shape=(1, count, 2560)))
    specs = {'x': xspec, **specs}
    report = author(output/'real', base, candidate, x, named, names, specs, small=False)
    bank_bytes = 512*640*2560*2
    report.update({'tokens': count, 'experts': 512, 'topK': 10, 'sourceFixture': str(fixture),
        'sourceManifestSHA256': sha256_file(fixture/'manifest.json'), 'weightSHA256': hashes,
        'weightBytes': sum(r['byteLength'] for r in records), 'learnedWeightBytesCopied': 0,
        'inputProvenance': f'First {count} tokens of prior deterministic S8192 activation; actual layer0 weights',
        'baseline': 'Native-copy I/O, integer grouping, NAX native-parity-v2 BM16 gate/up and BM32 down',
        'candidate': 'Same I/O/router/shared experts; per-weight FP16 dequant, BM32BN64 dense gate, up+parity activation, dense down',
        'denseBankBytesEach': bank_bytes, 'conservativeRetainedDenseBytes': bank_bytes*3,
        'orderedInputBytes': count*10*2560*2, 'gateOrActiveBytesEach': count*10*640*2,
        'orderedDownOrRestoredBytesEach': count*10*2560*2,
        'memoryCaution': 'Three 1.678GB dense banks may remain live:5.033GB plus activations/workspaces. Dependencies do not prove aliasing. Only one S4096 function in non-PLE GDN shared graph is a prospective integration scope; other graphs/counts retain packed kernels.',
        'largeCPUInferenceExecuted': False, 'qualityAccepted': False,
        'numericalCaution': 'Up rounding now matches measured native policy; full-K dense GEMM accumulation must still be compared on device.',
        'tolerances': {'maximumAbsoluteError': .002, 'relativeL2Error': .001}})
    write_json(output/'real/manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT/'results/coreai-prefill-1k/moe-dequant-s8192')
    parser.add_argument('--tokens', type=int, choices=(4096,), default=4096)
    parser.add_argument('--only', choices=('small', 'real', 'both'), default='both')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    hashes = {name: sha256_file(Path(__file__).with_name(name+'.py')) for name in SOURCES}
    result = {'status': 'CPU-authored-device-unvalidated', 'sourceSHA256': hashes}
    if args.only in ('small', 'both'):
        result['small'] = small_probe(output)
    if args.only in ('real', 'both'):
        result['real'] = real_probe(output, args.fixture.resolve(), args.tokens)
    if hashes != {name: sha256_file(Path(__file__).with_name(name+'.py')) for name in SOURCES}:
        raise RuntimeError('Authoring sources changed during export')
    write_json(output/'manifest.json', result)


if __name__ == '__main__':
    main()
