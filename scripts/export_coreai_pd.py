#!/usr/bin/env python3
"""Author fused CoreAI decoder layers with shared S1/decode and S4/prefill weights.

CPU export only. Full expert banks retain original affine Q4 bytes. The last-token
head avoids vocabulary projection for intermediate prompt positions. State is
explicit and identical across entrypoints; no padding can advance model state.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import shutil
import time
from types import MethodType
import torch
from export_moe import Source, sha256_file, write_json
from export_coreai_dense import DenseConfig, HCRead, HCWrite, PLE, Embedding, Head, read_hc
from export_coreai_hybrid import prepare_layer, state_metadata, atomic_manifest
from export_coreai_q4_moe import load_layer, PROJECTIONS

ROOT = Path(__file__).resolve().parents[1]

def stable_qsa_linear(self, x, name):
    from coreai_dense_metal import dense_linear
    return dense_linear(x, getattr(self, name.replace('.', '_')))

def install_stable_projections(module):
    """Optional identical lane reductions for S1/S4, without changing weights."""
    from coreai_dense_metal import dense_linear
    from export_coreai_gdn import GDN
    from export_coreai_qsa import QwenQSA
    from export_coreai_q4_moe import Q4MoE
    for child in module.modules():
        if isinstance(child, QwenQSA): child.linear = MethodType(stable_qsa_linear, child)
        elif isinstance(child, (GDN, Q4MoE, HCRead, PLE)): child.linear = dense_linear

class DecoderLayer(torch.nn.Module):
    def __init__(self, attention, attention_read, moe, moe_read, write, state_count, ple=None):
        super().__init__()
        self.attention, self.attention_read = attention, attention_read
        self.moe, self.moe_read, self.write = moe, moe_read, write
        self.state_count, self.ple = state_count, ple

    def forward(self, stream, *values):
        if self.ple is not None:
            embedding, states = values[0], values[1:]
            stream, next_ple = self.ple(stream, embedding, states[-1])
            states = states[:-1]
        else:
            states = values
        mixed, injection = self.attention_read(stream)
        result = self.attention(mixed, *states)
        stream = self.write(stream, result[0], injection)
        mixed, injection = self.moe_read(stream)
        output = self.moe(mixed)[0]
        stream = self.write(stream, output, injection)
        updated = result[1:1 + self.state_count]
        if self.ple is not None:
            updated = (*updated, next_ple)
        return (stream, *updated)

class LastHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
    def forward(self, stream):
        return self.head(stream[:, -1:, :])


def export_shared(module, examples, output_names, path, custom_kernels=()):
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    if custom_kernels:
        converter.register_custom_kernels(list(custom_kernels))
    names = list(next(iter(examples.values())))
    for name, inputs in examples.items():
        args = tuple(inputs.values())
        converter.add_pytorch_module(module, entrypoint_name=name, input_names=names, output_names=output_names,
            externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA,
                composite_op_name="scaled_dot_product_attention", composite_attrs=["scale", "is_causal", "window_size"])],
            export_fn=lambda m, args=args: torch.export.export(m, args=args).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    files = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(path.rglob('*')) if p.is_file()]
    return {"path": path.name, "function": "main", "prefillFunction": "prefill", "inputNames": names,
            "outputNames": list(output_names), "modelBytes": sum(p['bytes'] for p in files), "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=4, choices=(4, 8, 16))
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--q4-kernel', choices=('reference', 'metal'), default='reference',
                        help='reference expands selected weights; metal directly computes from packed Q4')
    parser.add_argument('--stable-projections', action='store_true',
                        help='Experimental fixed-reduction dense Metal kernels shared by S1/S4')
    parser.add_argument('--layers', help='Comma-separated layer subset for smoke; omit for all48')
    parser.add_argument('--components', nargs='+', choices=('layers','embedding','head'), default=['layers','embedding','head'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source = Source(0)
    config_path = source.directory / 'config.json'
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    layers = list(range(48)) if args.layers is None else [int(x) for x in args.layers.split(',')]
    if len(set(layers)) != len(layers) or any(x < 0 or x >= 48 for x in layers):
        raise ValueError('Invalid layer subset')
    if args.capacity < args.chunk or args.capacity > 4096 or args.capacity % 4:
        raise ValueError('Initial PD capacity must be a multiple of4 no larger than4096')
    if args.output.exists(): raise FileExistsError('Use a fresh output directory')
    estimated = len(layers) * 1_650_000_000 if 'layers' in args.components else 0
    ancestor = args.output.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < estimated + 4_000_000_000:
        raise ValueError('Insufficient space for full original-Q4 assets')
    args.output.mkdir(parents=True)
    manifest = {'version':1,'backend':'native-coreai-pd','status':'exporting','completeModelLayerSet':False,
        'capacity':args.capacity,'tokenChunk':args.chunk,'hiddenSize':c.hidden,'streamCount':c.streams,
        'vocabularySize':c.vocabulary,'modelDirectory':str(source.directory),'configSHA256':sha256_file(config_path),
        'assets':{},'layers':[], 'exporterSHA256':sha256_file(Path(__file__)), 'q4Kernel':args.q4_kernel,
        'stableProjections':args.stable_projections,
        'limitations':['CPU export only; full runtime numerics/performance require validation.',
            'No BF16 source quality-equivalence claim.']}
    custom_kernels = []
    if args.q4_kernel == 'metal':
        from coreai_q4_metal import MetalPackedQ4, get_q4_kernel
        custom_kernels = [get_q4_kernel()]
        manifest['q4KernelSHA256'] = sha256_file(Path(__file__).with_name('coreai_q4_metal.py'))
    else:
        manifest['limitations'].append('Reference Q4 selected-bank unpack remains unfused.')
    if args.stable_projections:
        from coreai_dense_metal import get_dense_kernel
        custom_kernels.append(get_dense_kernel())
        manifest['denseKernelSHA256'] = sha256_file(Path(__file__).with_name('coreai_dense_metal.py'))
    output = args.output / 'manifest.json'
    atomic_manifest(output, manifest)
    begin = time.perf_counter()
    try:
        if 'layers' in args.components:
            for layer in layers:
                started = time.perf_counter()
                source = Source(layer)
                moe = load_layer(source, 512, config['num_experts_per_tok'])
                if args.q4_kernel == 'metal':
                    for projection in PROJECTIONS:
                        setattr(moe, projection, MetalPackedQ4.from_packed(getattr(moe, projection)))
                kind, attention, states, bindings, _, _ = prepare_layer(source,config,layer,args.capacity)
                read_attention = HCRead(c,read_hc(source,f'language_model.model.layers.{layer}.attn_hyper_connection'))
                read_moe = HCRead(c,read_hc(source,f'language_model.model.layers.{layer}.mlp_hyper_connection'))
                ple = None
                states = dict(states)
                bindings = dict(bindings)
                state_count = len(states)
                if layer == 1:
                    source.prefix = 'language_model.model.layers.1.ple.'
                    ple = PLE(c,{name:source.read(name) for name in ('key_proj.weight','value_proj.weight',
                        'norm_key.weight','norm_query.weight','norm_conv.weight','conv1d.weight')})
                    states['ple_state'] = torch.zeros(1,c.ple_history,c.width,dtype=torch.float16)
                    bindings['ple_state'] = 'next_ple_state'
                module = DecoderLayer(attention,read_attention,moe,read_moe,HCWrite(c),state_count,ple).eval()
                if args.stable_projections: install_stable_projections(module)
                examples = {}
                for name, size in [('main',1),('prefill',args.chunk)]:
                    inputs = {'stream':torch.zeros(1,size,c.width,dtype=torch.float16)}
                    if ple is not None: inputs['ple_embedding'] = torch.zeros(1,size,c.ple_dim,dtype=torch.float16)
                    examples[name] = {**inputs,**states}
                asset = export_shared(module,examples,('stream_out',*bindings.values()),args.output/f'layer-{layer:02d}-{kind}.aimodel',custom_kernels)
                asset.update(index=layer,kind=kind,inputName='stream',outputName='stream_out',expertCount=512,
                    topK=10,completeExpertBank=True,stateBindings=bindings,initialState=state_metadata(states),hasPLE=ple is not None)
                manifest['layers'].append(asset)
                atomic_manifest(output,manifest)
                print(f"Layer {layer:02d} {kind}: {asset['modelBytes']} bytes in {time.perf_counter()-started:.2f}s",flush=True)
                del module,attention,read_attention,read_moe,moe,ple,states,examples,source
                gc.collect()
        for component in ('embedding','head'):
            if component not in args.components: continue
            source = Source(0)
            if component == 'embedding':
                source.prefix=''
                module=Embedding(c,source.read('language_model.model.embed_tokens.weight')).eval()
                examples={name:{'token':torch.zeros(size,dtype=torch.int32)} for name,size in [('main',1),('prefill',args.chunk)]}
                names=('stream',)
            else:
                hc=read_hc(source,'language_model.model.hyper_connection_mixer',with_injection=False)
                source.prefix=''
                module=LastHead(Head(c,hc,source.read('language_model.lm_head.weight'))).eval()
                examples={name:{'stream':torch.zeros(1,size,c.width,dtype=torch.float16)} for name,size in [('main',1),('prefill',args.chunk)]}
                names=('logits',)
            if args.stable_projections: install_stable_projections(module)
            asset=export_shared(module,examples,names,args.output/f'{component}.aimodel',custom_kernels)
            manifest['assets'][component]=asset
            atomic_manifest(output,manifest)
            print(f"Exported {component}: {asset['modelBytes']} bytes",flush=True)
            del module,examples,source
            gc.collect()
        manifest['status']='complete'
        manifest['completeModelLayerSet']=len(manifest['layers'])==48 and set(manifest['assets'])=={'embedding','head'}
        manifest['modelBytes']=sum(x['modelBytes'] for x in manifest['layers'])+sum(x['modelBytes'] for x in manifest['assets'].values())
        manifest['authoringSeconds']=time.perf_counter()-begin
        atomic_manifest(output,manifest)
    except Exception as error:
        manifest.update(status='failed',error=str(error))
        atomic_manifest(output,manifest)
        raise

if __name__ == '__main__': main()
