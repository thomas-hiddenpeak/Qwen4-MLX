#!/usr/bin/env python3
"""Write the explicit expert32/grouped-down prefill preset for a fresh plugin.

Checks build provenance, files and model config/index without loading a native
library or any model tensor. This is neither autotuning nor a performance test.
The runner still verifies the actual loaded ABI, symbols and stock MLX identity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from build_mlx_moe_gateup import EXPORTS, FUNCTIONS

PACKAGE = Path(__file__).resolve().parents[1]
PRESET = 'expert32-grouped-down'
GEOMETRY = {'hidden_size':2560, 'num_hidden_layers':48, 'num_experts':512,
            'num_experts_per_tok':10, 'moe_intermediate_size':640, 'shared_expert_intermediate_size':640}
NATIVE_SOURCES = ('moe_gateup_bridge.cpp','moe_gateup_bridge.h','moe_gateup_fused.metal','moe_expert_grouped.metal')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON number: ' + value)
    result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    require(type(result) is dict, 'Expected a JSON object')
    return result


def regular(path):
    path = Path(path).expanduser().resolve(strict=True)
    require(path.is_file(), 'Expected a regular file: ' + str(path))
    return path


def absolute(value):
    require(type(value) is str and Path(value).is_absolute(), 'Manifest paths must be absolute')
    return regular(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-fA-F]{64}', value) is not None, 'Invalid SHA256')
    return value.lower()


def checked_files(entries):
    require(type(entries) is list and entries, 'Missing manifest file inventory')
    result = {}
    for entry in entries:
        path = absolute(entry['path'])
        require(path not in result, 'Duplicate manifest file: ' + str(path))
        expected = digest(entry['sha256'])
        require(type(entry['bytes']) is int and entry['bytes'] >= 0 and path.stat().st_size == entry['bytes'], 'File size changed: ' + str(path))
        require(sha256(path) == expected, 'File SHA256 changed: ' + str(path))
        result[path] = expected
    return result


def prepare_configuration(model_dir, build_manifest, preset, *, package=None):
    package = (PACKAGE if package is None else Path(package)).resolve(strict=True)
    require(preset == PRESET, 'Unsupported preset')
    model = Path(model_dir).expanduser().resolve(strict=True)
    require(model.is_dir(), 'Model directory does not exist')
    manifest_path = regular(build_manifest)
    manifest_bytes = manifest_path.read_bytes()
    build = strict_json(manifest_bytes)
    require(type(build['schema_version']) is int and build['schema_version'] == 1 and build['status'] == 'built_not_executed'
            and build['original_inputs_unchanged'] is True and build.get('error') is None, 'Expected a completed unchanged isolated plugin build')
    require(type(build['abi_version']) is int and build['abi_version'] == 2 and set(build['exports']) == set(EXPORTS)
            and len(build['exports']) == len(EXPORTS) and set(build['functions']) == set(FUNCTIONS)
            and len(build['functions']) == len(FUNCTIONS), 'Build manifest does not declare the required ABI 2 exports/kernels')
    require(build['metal_fast_math'] is False and build['gpu_workload_started'] is False and build['library_loaded'] is False,
            'Expected the isolated no-fast-math build manifest, without library/model execution')
    require(build['commands'] and all(type(c['exit_code']) is int and c['exit_code'] == 0 for c in build['commands']), 'Native build command failed')
    runtime = Path(build['runtime'])
    output_root = Path(build['output_root'])
    require(runtime.is_absolute() and output_root.is_absolute(), 'Build roots must be absolute')
    runtime = runtime.resolve(strict=True); output_root = output_root.resolve(strict=True)
    require(runtime.is_dir() and output_root.is_dir() and runtime not in output_root.parents and output_root != runtime
            and manifest_path == output_root/'build-provenance.json', 'Manifest location/build isolation differs')
    library = absolute(build['library_path']); metallib = absolute(build['metallib_path'])
    require(library == output_root/'lib/libanemlx_moe_gateup.dylib' and metallib == output_root/'lib/moe_gateup.metallib', 'Native output paths differ from the build layout')
    originals = checked_files(build['original_inputs']); artifacts = checked_files(build['artifacts'])
    expected_artifacts = {library, metallib, output_root/'obj/moe_gateup_bridge.o', output_root/'obj/moe_gateup_fused.air', output_root/'obj/moe_expert_grouped.air'}
    require(set(artifacts) == expected_artifacts, 'Unexpected or missing native build artifacts')
    stock = regular(runtime/'lib/mlx/lib/libmlx.dylib'); stamp = regular(runtime/'lib/mlx/.version')
    require(stock in originals and stamp in originals and stamp.read_text().strip() == build['pinned_stamp'], 'Pinned base MLX/stamp missing or changed')
    equivalence = build['stock_install_equivalence']
    require(equivalence['exact_match'] is True and digest(equivalence['installed_sha256']) == originals[stock], 'Build does not identify the exact stock MLX installation')
    # Match the CLI loader's package-default stock check without loading libmlx.
    package_stock = regular(package.parent/'qwen38-ssd/runtime/mlx-serve/lib/mlx/lib/libmlx.dylib')
    require(sha256(package_stock) == originals[stock], 'Build base MLX differs from this package\'s stock runtime')
    required_sources = [regular(package/'scripts/build_mlx_moe_gateup.py')] + [regular(package/'native'/name) for name in NATIVE_SOURCES]
    for source in required_sources:
        matches = [p for p in originals if os.path.samefile(p, source)]
        require(len(matches) == 1, 'Build provenance lacks the current builder/native source: ' + str(source))
    for name in NATIVE_SOURCES:
        copied = regular(output_root/'src'/name)
        require(sha256(copied) == sha256(package/'native'/name), 'Copied native source differs from this checkout: ' + name)
    config_path = regular(model/'config.json'); index_path = regular(model/'model.safetensors.index.json')
    config_bytes = config_path.read_bytes(); index_bytes = index_path.read_bytes()
    config = strict_json(config_bytes); index = strict_json(index_bytes)
    require(config['model_type'] == 'qwen4_exp' and type(config['text_config']) is dict, 'Expected qwen4_exp with text_config')
    text = config['text_config']; quant = config['quantization']
    require(all(type(text[k]) is int and text[k] == v for k,v in GEOMETRY.items())
            and text['hidden_act'] == 'silu' and text['output_gate_type'] == 'sigmoid', 'Model MoE geometry/activation is incompatible with expert32')
    require(quant['mode'] == 'affine' and type(quant['bits']) is int and quant['bits'] == 4
            and type(quant['group_size']) is int and quant['group_size'] == 64, 'Expected affine Q4/group64 quantization')
    weight_map = index['weight_map']
    require(type(weight_map) is dict and weight_map, 'Missing safetensors weight_map')
    for shard in set(weight_map.values()):
        require(type(shard) is str and shard == Path(shard).name and shard.endswith('.safetensors') and not shard.startswith('.')
                and '\\' not in shard, 'Unsafe indexed shard name')
        resolved = regular(model/shard)
        require(resolved.parent == model, 'Indexed shard resolves outside the model directory')
    # Validate required names and files; tensor shapes/bounds remain the runner's
    # responsibility. Do not open safetensors or read checkpoint payload here.
    required = []
    for layer in range(48):
        prefix = f'language_model.model.layers.{layer}.mlp.'
        required += [prefix+'gate.weight', prefix+'shared_expert_gate.weight']
        required += [prefix+'shared_expert.'+projection+'.weight' for projection in ('gate_proj','up_proj','down_proj')]
        required += [prefix+'switch_mlp.'+projection+'.'+part for projection in ('gate_proj','up_proj','down_proj') for part in ('weight','scales','biases')]
    require(all(name in weight_map for name in required), 'Model index lacks required routed/shared MoE tensors')
    # Recheck the inputs whose identity is embedded in the new selection before
    # returning. The runtime independently checks loaded libraries when used.
    require(manifest_path.read_bytes() == manifest_bytes and config_path.read_bytes() == config_bytes and index_path.read_bytes() == index_bytes
            and sha256(library) == artifacts[library] and sha256(metallib) == artifacts[metallib] and sha256(stock) == originals[stock], 'Identity changed while preparing configuration')
    return {'version':1, 'threadgroups':{}, 'gateUpVariant':2, 'groupedDown':True,
        'model_directory':str(model), 'gateup_plugin_path':str(library), 'gateup_plugin_sha256':artifacts[library],
        'base_mlx_path':str(stock), 'base_mlx_sha256':originals[stock], 'native_configuration':0,
        'preset':PRESET, 'status':'explicit_preset_not_autotuned_or_benchmarked',
        'build_manifest_path':str(manifest_path), 'build_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
        'metallib_path':str(metallib), 'metallib_sha256':artifacts[metallib], 'declared_plugin_abi_version':2,
        'model_metadata_sha256':{'config.json':hashlib.sha256(config_bytes).hexdigest(), 'model.safetensors.index.json':hashlib.sha256(index_bytes).hexdigest()},
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'validation_scope':'Offline build/source/artifact hashes and model config/index only. No dlopen, tensor loading, numerical test, autotune or performance measurement. Runner verifies actual ABI, symbol ownership, loaded stock MLX and tensor layouts.',
        'runtime_requirements':{'ANERUNNER_GATEUP_LIBRARY':str(library),'prefill_accumulation':'reference',
            'ANERUNNER_MOE_QMM_CONFIG':'unset or 0','ANERUNNER_MOE_QMM_BM':'unset or 0'},
        'selection_scope':'Explicit request-local prefill: expert32 gate/up and grouped down for chunks 205...512. Original reduction; other lengths, decode and verification retain their existing paths.'}


def write_new(path, configuration):
    path = Path(path).expanduser()
    require(not os.path.lexists(path), 'Refuse to overwrite output: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(configuration, stream, indent=2, allow_nan=False); stream.write('\n')
    return path.resolve()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--build-manifest', type=Path, required=True)
    parser.add_argument('--preset', choices=[PRESET], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(not os.path.lexists(args.output.expanduser()), 'Refuse to overwrite output: ' + str(args.output))
        configuration = prepare_configuration(args.model_dir, args.build_manifest, args.preset)
        output = write_new(args.output, configuration)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f'error: {error}\n')
    print(json.dumps({'output':str(output), 'preset':PRESET, 'status':configuration['status'],
                      'ANERUNNER_GATEUP_LIBRARY':configuration['gateup_plugin_path']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
