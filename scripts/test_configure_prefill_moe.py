"""Synthetic CPU controls for the explicit prefill preset; no native loading."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys
sys.dont_write_bytecode = True
import configure_prefill_moe as m


class ConfigurePrefillMoEControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='prefill-preset-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root/'ane-runner'; self.package.mkdir()
        self.runtime = self.root/'qwen38-ssd/runtime/mlx-serve'
        self.native = self.root/'new-native'; self.native.mkdir()
        self.model = self.root/'model'; self.model.mkdir()
        def put(path, data):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path
        self.put = put
        self.stock = put(self.runtime/'lib/mlx/lib/libmlx.dylib', b'synthetic stock bytes')
        stamp = put(self.runtime/'lib/mlx/.version', b'synthetic pinned stamp\n')
        builder = put(self.package/'scripts/build_mlx_moe_gateup.py', b'synthetic builder source')
        originals = [self.stock, stamp, builder]
        for name in m.NATIVE_SOURCES:
            data = ('synthetic source: '+name).encode()
            originals.append(put(self.package/'native'/name, data))
            put(self.native/'src'/name, data)
        self.library = put(self.native/'lib/libanemlx_moe_gateup.dylib', b'synthetic plugin bytes')
        self.metallib = put(self.native/'lib/moe_gateup.metallib', b'synthetic metallib bytes')
        artifacts = [self.library,self.metallib]
        for name in ('moe_gateup_bridge.o','moe_gateup_fused.air','moe_expert_grouped.air'):
            artifacts.append(put(self.native/'obj'/name, ('synthetic '+name).encode()))
        def entry(path): return {'path':str(path),'sha256':m.sha256(path),'bytes':path.stat().st_size}
        self.manifest = {'schema_version':1,'status':'built_not_executed','original_inputs_unchanged':True,
            'abi_version':2,'exports':list(m.EXPORTS),'functions':list(m.FUNCTIONS),
            'metal_fast_math':False,'gpu_workload_started':False,'library_loaded':False,
            'commands':[{'exit_code':0}],'runtime':str(self.runtime),'output_root':str(self.native),
            'library_path':str(self.library),'metallib_path':str(self.metallib),
            'original_inputs':[entry(p) for p in originals],'artifacts':[entry(p) for p in artifacts],
            'pinned_stamp':'synthetic pinned stamp','stock_install_equivalence':{'exact_match':True,'installed_sha256':m.sha256(self.stock)}}
        self.manifest_path = self.native/'build-provenance.json'; self.save_manifest()
        self.config = {'model_type':'qwen4_exp', 'text_config':dict(m.GEOMETRY,hidden_act='silu',output_gate_type='sigmoid'),
            'quantization':{'mode':'affine','bits':4,'group_size':64}}
        self.save_config()
        weight_map={}
        for layer in range(48):
            prefix=f'language_model.model.layers.{layer}.mlp.'
            for name in ('gate.weight','shared_expert_gate.weight'):
                weight_map[prefix+name]='model.safetensors'
            for projection in ('gate_proj','up_proj','down_proj'):
                weight_map[prefix+'shared_expert.'+projection+'.weight']='model.safetensors'
                for part in ('weight','scales','biases'):
                    weight_map[prefix+'switch_mlp.'+projection+'.'+part]='model.safetensors'
        self.index={'weight_map':weight_map}
        self.save_index()
        put(self.model/'model.safetensors', b'never opened: this is not a real tensor file')

    def save_manifest(self): self.manifest_path.write_text(json.dumps(self.manifest))
    def save_config(self): (self.model/'config.json').write_text(json.dumps(self.config))
    def save_index(self): (self.model/'model.safetensors.index.json').write_text(json.dumps(self.index))
    def prepare(self): return m.prepare_configuration(self.model,self.manifest_path,m.PRESET,package=self.package)

    def test_loader_contract_hashes_without_tensor_or_process_execution(self):
        original_open = Path.open
        def file_open(path, *args, **kwargs):
            if path.suffix == '.safetensors': raise AssertionError('Tensor file must not be opened')
            return original_open(path,*args,**kwargs)
        with mock.patch.object(Path,'open',file_open), mock.patch('subprocess.run',side_effect=AssertionError('No external command')):
            result=self.prepare()
        self.assertEqual({k:result[k] for k in ('version','threadgroups','gateUpVariant','groupedDown')},
                         {'version':1,'threadgroups':{},'gateUpVariant':2,'groupedDown':True})
        self.assertEqual(result['model_directory'],str(self.model.resolve()))
        self.assertEqual(result['gateup_plugin_sha256'],m.sha256(self.library))
        self.assertEqual(result['base_mlx_sha256'],m.sha256(self.stock))
        self.assertEqual(result['declared_plugin_abi_version'],2)
        self.assertEqual(result['status'],'explicit_preset_not_autotuned_or_benchmarked')
        self.assertEqual(result['model_metadata_sha256']['config.json'],m.sha256(self.model/'config.json'))
        self.assertEqual(result['runtime_requirements']['ANERUNNER_GATEUP_LIBRARY'],str(self.library.resolve()))

    def test_changed_native_artifact_hash_rejected(self):
        original=self.library.read_bytes(); self.library.write_bytes(b'X'+original[1:])
        with self.assertRaisesRegex(ValueError,'SHA256 changed'): self.prepare()

    def test_abi_and_export_contract_rejected(self):
        for field,value in [('abi_version',1),('exports',list(m.EXPORTS)[:-1]),('metal_fast_math',True)]:
            with self.subTest(field=field):
                old=self.manifest[field];self.manifest[field]=value;self.save_manifest()
                with self.assertRaises(ValueError):self.prepare()
                self.manifest[field]=old;self.save_manifest()

    def test_wrong_model_geometry_or_quantization_rejected(self):
        for section,key,value in [('text_config','hidden_size',2048),('text_config','num_experts',True),('quantization','group_size',128)]:
            with self.subTest(key=key):
                old=self.config[section][key];self.config[section][key]=value;self.save_config()
                with self.assertRaises(ValueError):self.prepare()
                self.config[section][key]=old;self.save_config()

    def test_missing_indexed_expert_rejected(self):
        del self.index['weight_map']['language_model.model.layers.47.mlp.switch_mlp.down_proj.biases'];self.save_index()
        with self.assertRaisesRegex(ValueError,'lacks required'):self.prepare()

    def test_unsafe_shard_and_external_symlink_rejected(self):
        key=next(iter(self.index['weight_map']));self.index['weight_map'][key]='../outside.safetensors';self.save_index()
        with self.assertRaisesRegex(ValueError,'Unsafe'):self.prepare()
        self.index['weight_map'][key]='external.safetensors';self.save_index()
        outside=self.put(self.root/'outside.safetensors',b'outside')
        (self.model/'external.safetensors').symlink_to(outside)
        with self.assertRaisesRegex(ValueError,'outside the model'):self.prepare()

    def test_wrong_package_stock_rejected(self):
        # Build may point elsewhere, but the selection loader compares its own stock.
        elsewhere=self.root/'other-runtime';other_stock=self.put(elsewhere/'lib/mlx/lib/libmlx.dylib',self.stock.read_bytes())
        other_stamp=self.put(elsewhere/'lib/mlx/.version',b'synthetic pinned stamp\n')
        for entry in self.manifest['original_inputs']:
            if entry['path']==str(self.stock):entry['path']=str(other_stock)
            elif entry['path']==str(self.runtime/'lib/mlx/.version'):entry['path']=str(other_stamp)
        self.manifest['runtime']=str(elsewhere);self.save_manifest()
        self.stock.write_bytes(b'different package stock')
        with self.assertRaisesRegex(ValueError,'package.*stock runtime'):self.prepare()

    def test_existing_and_dangling_output_are_not_overwritten(self):
        result=self.prepare();output=self.root/'config.json';output.write_text('keep')
        with self.assertRaisesRegex(ValueError,'overwrite'):m.write_new(output,result)
        self.assertEqual(output.read_text(),'keep')
        dangling=self.root/'dangling.json';dangling.symlink_to(self.root/'missing.json')
        with self.assertRaisesRegex(ValueError,'overwrite'):m.write_new(dangling,result)
        self.assertFalse((self.root/'missing.json').exists())

    def test_duplicate_manifest_json_rejected(self):
        self.manifest_path.write_text('{"status":"failed","status":"built_not_executed"}')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON'):self.prepare()

    def test_actual_cli_new_file_and_no_overwrite(self):
        output=self.root/'output/preset.json'
        args=['--model-dir',str(self.model),'--build-manifest',str(self.manifest_path),'--preset',m.PRESET,'--output',str(output)]
        with mock.patch.object(m,'PACKAGE',self.package), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(m.main(args),0)
        saved=json.loads(output.read_text());self.assertEqual(saved['gateUpVariant'],2)
        original=output.read_bytes()
        with mock.patch.object(m,'PACKAGE',self.package), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as exit_result:m.main(args)
        self.assertEqual(exit_result.exception.code,1);self.assertEqual(output.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
