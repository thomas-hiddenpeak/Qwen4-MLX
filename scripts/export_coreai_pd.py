#!/usr/bin/env python3
"""Author fused CoreAI decoder layers with shared decode and fixed-size prefill weights.

CPU export only. Full expert banks retain original affine Q4 bytes. The last-token
head avoids vocabulary projection for intermediate prompt positions. State is
explicit and identical across entrypoints; no padding can advance model state.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
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


def authoring_source_hashes():
    """Pin imported authoring math as well as the independently tracked kernels."""
    names = ('export_moe', 'export_coreai_dense', 'export_coreai_hybrid',
             'export_coreai_gdn', 'export_coreai_qsa', 'export_coreai_q4_moe',
             'coreai_q4_metal', 'coreai_dense_metal', 'coreai_tensor_matmul',
             'coreai_gdn_chunk', 'coreai_gdn_chunk_metal', 'coreai_moe_chunk',
             'coreai_q4_grouped', 'coreai_q4_gateup', 'coreai_qsa_chunk')
    return {name: sha256_file(Path(__file__).with_name(name + '.py')) for name in names}


def source_provenance_hash(source):
    """Identity of Source's already verified shard ledger, not a new shard scan."""
    identity = {'sourceManifest': source.manifest, 'weightIndex': source.index,
                'verifiedFiles': source.verified}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_recorded_asset(output, asset):
    """Validate the exact recorded file set before trusting a completed asset."""
    name = asset.get('path')
    if not isinstance(name, str) or Path(name).name != name or not name.endswith('.aimodel'):
        raise ValueError(f'Invalid recorded asset path: {name!r}')
    directory = output / name
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f'Recorded asset is missing or is a symlink: {directory}')
    records = asset.get('files', [])
    if not records or len({record['path'] for record in records}) != len(records):
        raise ValueError(f'Missing or duplicate file records: {directory}')
    paths = list(directory.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError(f'Symlinks are unsupported in a recorded asset: {directory}')
    actual = {str(path.relative_to(directory)) for path in paths if path.is_file()}
    if actual != {record['path'] for record in records}:
        raise ValueError(f'Asset file set differs from its manifest: {directory}')
    total = 0
    for record in records:
        relative = Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid asset file path: {record["path"]!r}')
        path = directory / relative
        if path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']:
            raise ValueError(f'Asset size/SHA256 mismatch: {path}')
        total += record['bytes']
    if total != asset.get('modelBytes'):
        raise ValueError(f'Asset byte total differs from its manifest: {directory}')


def resume_manifest(output, expected, *, allow_exporter_change=False, layer_kinds=None):
    """Return a validated resumed manifest; never overwrite or remove an asset.

    Pre-resume manifests lack request/dependency ledgers. Their explicit upgrade
    accepts only a full-model request, compares every hash they did record, and
    records which additional metadata starts at this resume boundary.
    """
    path = output / 'manifest.json'
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get('status') != 'exporting' or manifest.get('completeModelLayerSet') is not False:
        raise ValueError('Resume requires an incomplete manifest with status=exporting')
    prior_hash = manifest.get('exporterSHA256')
    current_hash = expected['exporterSHA256']
    if not isinstance(prior_hash, str) or len(prior_hash) != 64 or any(c not in '0123456789abcdef' for c in prior_hash):
        raise ValueError('Resume manifest has no valid prior exporter SHA256')
    if prior_hash != current_hash and not allow_exporter_change:
        raise ValueError('Exporter SHA256 changed; review the change and pass --resume-exporter-change explicitly')
    upgrade_keys = {'requestedLayers', 'requestedComponents', 'authoringSourceSHA256', 'sourceProvenanceSHA256'}
    added = sorted(upgrade_keys - manifest.keys())
    if added and (not allow_exporter_change or expected['requestedLayers'] != list(range(48))
                  or set(expected['requestedComponents']) != {'layers', 'embedding', 'head'}):
        raise ValueError('Legacy resume metadata requires --resume-exporter-change and the default full-model request')
    # Compare all recorded numerical/configuration/source choices, including
    # nested kernel hashes. The exporter exception never permits kernel drift.
    ignored = {'status', 'completeModelLayerSet', 'layers', 'assets', 'exporterSHA256', 'limitations'}
    for key, value in expected.items():
        if key in ignored or key in added:
            continue
        if key not in manifest or manifest[key] != value:
            raise ValueError(f'Resume configuration/source mismatch: {key}')
    layers = manifest.get('layers', [])
    assets = manifest.get('assets', {})
    requested = set(expected['requestedLayers']) if 'layers' in expected['requestedComponents'] else set()
    indices = [asset.get('index') for asset in layers]
    if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices) or not set(indices) <= requested:
        raise ValueError('Resume manifest has duplicate or unrequested layer indices')
    if not set(assets) <= set(expected['requestedComponents']) - {'layers'}:
        raise ValueError('Resume manifest has unrequested components')
    names = []
    for asset in layers:
        index, kind = asset['index'], asset.get('kind')
        if kind not in ('gdn', 'qsa') or (layer_kinds is not None and kind != layer_kinds[index]):
            raise ValueError(f'Recorded layer {index} has the wrong attention kind')
        if asset.get('path') != f'layer-{index:02d}-{kind}.aimodel':
            raise ValueError(f'Recorded layer {index} has the wrong asset path')
    for component, asset in assets.items():
        if asset.get('path') != f'{component}.aimodel':
            raise ValueError(f'Recorded component {component} has the wrong asset path')
    for asset in [*layers, *assets.values()]:
        if asset.get('function') != 'main' or asset.get('prefillFunction') != 'prefill':
            raise ValueError('Recorded asset has unexpected phase functions')
        names.append(asset['path'])
        print(f"Verifying completed asset: {asset['path']}", flush=True)
        verify_recorded_asset(output, asset)
    if len(set(names)) != len(names):
        raise ValueError('Duplicate recorded asset paths')
    unrecorded = sorted(str(path) for path in output.glob('*.aimodel') if path.name not in names)
    if unrecorded:
        raise FileExistsError('Unrecorded/incomplete asset paths; move these aside before resuming (nothing was removed): '
                              + ', '.join(unrecorded))
    if path.read_bytes() != raw:
        raise ValueError('Manifest changed during resume validation; stop the active exporter before resuming')
    for key in added:
        manifest[key] = expected[key]
    manifest.setdefault('originalExporterSHA256', prior_hash)
    manifest['exporterSHA256'] = current_hash
    manifest.setdefault('resumeHistory', []).append({
        'at': datetime.now(timezone.utc).isoformat(),
        'previousExporterSHA256': prior_hash, 'currentExporterSHA256': current_hash,
        'exporterChangeExplicitlyAllowed': allow_exporter_change,
        'previousManifestSHA256': hashlib.sha256(raw).hexdigest(),
        'legacyMetadataAddedAtResume': added,
        'validatedLayerIndices': sorted(indices), 'validatedComponents': sorted(assets),
    })
    return manifest

def phase_linear(x, weight):
    """Large prompts use TensorOps; S1 retains the existing dense decode path."""
    if x.shape[1] == 1:
        return torch.nn.functional.linear(x.float(), weight.float()).half()
    from coreai_tensor_matmul import tensor_linear
    return tensor_linear(x, weight)


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
    # A nonempty unmatched spec still makes the SDK export every entrypoint
    # twice. GDN/embedding/head have no SDPA and need no externalization pass.
    externalize = [coreai_torch.ExternalizeSpec(target_class=SDPA,
        composite_op_name="scaled_dot_product_attention", composite_attrs=["scale", "is_causal", "window_size"])] \
        if any(isinstance(child, SDPA) for child in module.modules()) else None
    names = list(next(iter(examples.values())))
    for name, inputs in examples.items():
        args = tuple(inputs.values())
        converter.add_pytorch_module(module, entrypoint_name=name, input_names=names, output_names=output_names,
            externalize_modules=externalize,
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
    parser.add_argument('--resume', action='store_true', help='Verify and continue an existing status=exporting manifest; stop its previous exporter first')
    parser.add_argument('--resume-exporter-change', action='store_true', help='Explicitly allow only exporter-file changes and record their hash lineage')
    parser.add_argument('--chunk', type=int, default=4, choices=(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048))
    parser.add_argument('--tail-chunks', type=int, nargs='*', default=None,
                        help='Optional smaller prefill entrypoints; tensor mode defaults to 4/16/64 below main size')
    parser.add_argument('--prefill-kernels', choices=('reference','tensor'), default='reference',
                        help='Experimental batched TensorOps, grouped Q4, register GDN and incremental QSA pool')
    parser.add_argument('--prefill-sdpa', choices=('float32','float16'), default='float32',
                        help='Tensor prefill SDPA operand type; decode remains float32')
    parser.add_argument('--fused-gateup', action='store_true', help='Fuse grouped expert gate/up and activation in tensor prefill')
    parser.add_argument('--capacity', type=int, default=4096)
    parser.add_argument('--q4-kernel', choices=('reference', 'metal'), default='reference',
                        help='reference expands selected weights; metal directly computes from packed Q4')
    parser.add_argument('--stable-projections', action='store_true',
                        help='Experimental fixed-reduction dense Metal kernels shared by S1/S4')
    parser.add_argument('--layers', help='Comma-separated layer subset for smoke; omit for all48')
    parser.add_argument('--components', nargs='+', choices=('layers','embedding','head'), default=['layers','embedding','head'])
    args = parser.parse_args()
    if args.resume_exporter_change and not args.resume:
        raise ValueError('--resume-exporter-change requires --resume')
    if len(set(args.components)) != len(args.components):
        raise ValueError('Duplicate components are unsupported')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source = Source(0)
    config_path = source.directory / 'config.json'
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    layers = list(range(48)) if args.layers is None else [int(x) for x in args.layers.split(',')]
    if len(set(layers)) != len(layers) or any(x < 0 or x >= 48 for x in layers):
        raise ValueError('Invalid layer subset')
    tails = args.tail_chunks if args.tail_chunks is not None else ([s for s in (4,8,16,32,64,128,256,512,1024) if s < args.chunk] if args.prefill_kernels == 'tensor' else [])
    if len(set(tails)) != len(tails) or any(s not in (4,8,16,32,64,128,256,512,1024) or s >= args.chunk for s in tails):
        raise ValueError('Tail chunks must be unique supported sizes below the primary chunk')
    if args.prefill_kernels == 'tensor' and (args.q4_kernel != 'metal' or args.stable_projections):
        raise ValueError('Tensor prefill requires packed Metal Q4 and excludes stable GEMV projections')
    if args.prefill_kernels != 'tensor' and (args.prefill_sdpa != 'float32' or args.fused_gateup):
        raise ValueError('SDPA precision and grouped fusion require tensor prefill')
    if args.chunk > 16 and args.prefill_kernels != 'tensor':
        raise ValueError('Large chunks require tensor prefill kernels')
    if args.capacity < args.chunk or args.capacity > 16384 or args.capacity % 4:
        raise ValueError('PD capacity must be a multiple of4 no larger than16384')
    phase_sizes = [('main',1),('prefill',args.chunk)] + [(f'prefill_s{s}',s) for s in sorted(tails)]
    if args.output.exists() and not args.resume:
        raise FileExistsError('Use a fresh output directory or explicitly pass --resume')
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError('Resume output directory does not exist')
    manifest = {'version':1,'backend':'native-coreai-pd','status':'exporting','completeModelLayerSet':False,
        'capacity':args.capacity,'tokenChunk':args.chunk,'tailChunks':sorted(tails),'prefillKernels':args.prefill_kernels,'prefillSDPA':args.prefill_sdpa,'fusedGateUp':args.fused_gateup,'moeTile':[16,32,64],'hiddenSize':c.hidden,'streamCount':c.streams,
        'vocabularySize':c.vocabulary,'modelDirectory':str(source.directory),'configSHA256':sha256_file(config_path),
        'assets':{},'layers':[], 'exporterSHA256':sha256_file(Path(__file__)), 'q4Kernel':args.q4_kernel,
        'stableProjections':args.stable_projections,
        'requestedLayers':sorted(layers),'requestedComponents':sorted(args.components),
        'authoringSourceSHA256':authoring_source_hashes(),'sourceProvenanceSHA256':source_provenance_hash(source),
        'limitations':['CPU export only; full runtime numerics/performance require validation.',
            'No BF16 source quality-equivalence claim.']}
    custom_kernels = []
    if args.q4_kernel == 'metal':
        from coreai_q4_metal import MetalPackedQ4, get_q4_kernel
        custom_kernels = [get_q4_kernel()]
        manifest['q4KernelSHA256'] = sha256_file(Path(__file__).with_name('coreai_q4_metal.py'))
    else:
        manifest['limitations'].append('Reference Q4 selected-bank unpack remains unfused.')
    if args.prefill_kernels == 'tensor':
        from coreai_tensor_matmul import get_tensor_kernel
        from coreai_gdn_chunk_metal import get_gdn_recurrence_kernel
        from coreai_q4_grouped import get_plan_kernel, get_grouped_kernel
        from coreai_moe_chunk import ChunkQ4MoE
        from coreai_gdn_chunk import GDNRegisterPrefill
        from coreai_qsa_chunk import QwenQSAChunk, get_pool_kernel
        custom_kernels += [get_tensor_kernel(),get_gdn_recurrence_kernel(),get_plan_kernel(),get_grouped_kernel(16,32,64)]
        if args.fused_gateup:
            from coreai_q4_gateup import get_gateup_kernel
            custom_kernels.append(get_gateup_kernel(16,32,64))
        manifest['prefillKernelSHA256'] = {name:sha256_file(Path(__file__).with_name(name+'.py')) for name in
            ('coreai_tensor_matmul','coreai_gdn_chunk','coreai_gdn_chunk_metal','coreai_moe_chunk','coreai_q4_grouped','coreai_q4_gateup','coreai_qsa_chunk')}
    if args.stable_projections:
        from coreai_dense_metal import get_dense_kernel
        custom_kernels.append(get_dense_kernel())
        manifest['denseKernelSHA256'] = sha256_file(Path(__file__).with_name('coreai_dense_metal.py'))
    output = args.output / 'manifest.json'
    if args.resume:
        layer_kinds = ['gdn' if kind == 'linear_attention' else 'qsa' for kind in config['layer_types']]
        manifest = resume_manifest(args.output, manifest, allow_exporter_change=args.resume_exporter_change,
                                   layer_kinds=layer_kinds)
    completed_layers = {asset['index'] for asset in manifest['layers']}
    estimated = len(set(layers) - completed_layers) * 1_650_000_000 if 'layers' in args.components else 0
    ancestor = args.output if args.output.exists() else args.output.parent
    while not ancestor.exists(): ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < estimated + 4_000_000_000:
        raise ValueError('Insufficient space for remaining original-Q4 assets')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    atomic_manifest(output, manifest)
    begin = time.perf_counter()
    try:
        if 'layers' in args.components:
            for layer in layers:
                if layer in completed_layers:
                    print(f'Skipping verified layer {layer:02d}', flush=True)
                    continue
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
                layer_kernels = list(custom_kernels)
                if args.prefill_kernels == 'tensor':
                    moe = ChunkQ4MoE(moe, columns=32, inner=64, fuse_gateup=args.fused_gateup)
                    if kind == 'gdn':
                        attention.linear = phase_linear
                        attention = GDNRegisterPrefill(attention)
                    else:
                        attention = QwenQSAChunk(attention, prefill_sdpa_fp16=args.prefill_sdpa == 'float16')
                        layer_kernels += attention.custom_kernels()[1:]
                    read_attention.linear = phase_linear
                    read_moe.linear = phase_linear
                    if ple is not None: ple.linear = phase_linear
                module = DecoderLayer(attention,read_attention,moe,read_moe,HCWrite(c),state_count,ple).eval()
                if args.stable_projections: install_stable_projections(module)
                examples = {}
                for name, size in phase_sizes:
                    inputs = {'stream':torch.zeros(1,size,c.width,dtype=torch.float16)}
                    if ple is not None: inputs['ple_embedding'] = torch.zeros(1,size,c.ple_dim,dtype=torch.float16)
                    examples[name] = {**inputs,**states}
                asset = export_shared(module,examples,('stream_out',*bindings.values()),args.output/f'layer-{layer:02d}-{kind}.aimodel',layer_kernels)
                asset.update(index=layer,kind=kind,inputName='stream',outputName='stream_out',expertCount=512,
                    topK=10,completeExpertBank=True,stateBindings=bindings,initialState=state_metadata(states),hasPLE=ple is not None)
                manifest['layers'].append(asset)
                atomic_manifest(output,manifest)
                print(f"Layer {layer:02d} {kind}: {asset['modelBytes']} bytes in {time.perf_counter()-started:.2f}s",flush=True)
                del module,attention,read_attention,read_moe,moe,ple,states,examples,source
                gc.collect()
        for component in ('embedding','head'):
            if component not in args.components: continue
            if component in manifest['assets']:
                print(f'Skipping verified component {component}', flush=True)
                continue
            source = Source(0)
            if component == 'embedding':
                source.prefix=''
                module=Embedding(c,source.read('language_model.model.embed_tokens.weight')).eval()
                examples={name:{'token':torch.zeros(size,dtype=torch.int32)} for name,size in phase_sizes}
                names=('stream',)
            else:
                hc=read_hc(source,'language_model.model.hyper_connection_mixer',with_injection=False)
                source.prefix=''
                module=LastHead(Head(c,hc,source.read('language_model.lm_head.weight'))).eval()
                examples={name:{'stream':torch.zeros(1,size,c.width,dtype=torch.float16)} for name,size in phase_sizes}
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
