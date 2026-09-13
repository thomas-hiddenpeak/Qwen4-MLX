#!/usr/bin/env python3
"""Prepare one CPU-native P262112/O32 fixture, or validate its raw provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PREFIX = 'This is a synthetic long-context cache capacity test. Ignore the filler words below.\n'
TAIL = '\nNow write the integers from 1 to 100, separated by commas. Start with 1,2,3 and do not explain.'
UNIT = ' record'
DECODE_TAIL = '\nWrite 1 to 100, separated by commas. Start with 1,2,3.'
FILES = ('prompt.txt', 'tokenization.json', 'tokens.json', 'request.json', 'tokenizer.log')


def require(okay, message):
    if not okay:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def base_long(root, base):
    require(base.get('schema') == 'qwen-long-http-fixture-v1' and base.get('complete') is True
            and base.get('context_limit') == 262144, 'Extension requires the complete262144 base fixture')
    record = base['tokenizations']['long']
    for kind in ('prompt', 'tokenization', 'tokens'):
        require(sha(root / record[kind+'_file']) == record[kind+'_sha256'], 'Base long tokenizer artifact changed')
    request_record = base['requests']['long']
    require(sha(root/request_record['file']) == request_record['sha256'], 'Base long request changed')
    prompt = (root/record['prompt_file']).read_text()
    ids = json.loads((root/record['tokens_file']).read_text())
    native = json.loads((root/record['tokenization_file']).read_text())
    request = json.loads((root/request_record['file']).read_text())
    require(ids == native['tokens'] and len(ids) == 262142, 'Base native prompt count differs')
    require(native['decoded'] == native['rendered_prompt'] and native['rendered_prompt'].count(prompt.strip()) == 1,
            'Base native rendering/roundtrip differs')
    require(request['messages'] == [{'role':'user','content':prompt}] and request['max_tokens'] == 2,
            'Base native prompt/HTTP message binding differs')
    require(prompt.startswith(PREFIX) and prompt.endswith(TAIL), 'Base prompt is not the calibrated authored fixture')
    middle = prompt[len(PREFIX):-len(TAIL)]
    repeats = len(middle)//len(UNIT)
    require(repeats > 30 and middle == UNIT*repeats, 'Base filler is not an exact UNIT repetition')
    return PREFIX+UNIT*(repeats-22)+DECODE_TAIL, ids


def load_decode_fixture(directory, base_root, base, runner):
    directory, base_root = Path(directory), Path(base_root)
    f = json.loads((directory/'fixture.json').read_text())
    require(f.get('schema') == 'qwen-long-http-decode-fixture-v1' and f.get('complete') is True,
            'Extended decode fixture is incomplete')
    require(f.get('base_fixture_sha256') == sha(base_root/'fixture.json'), 'Extension bound to a different base fixture')
    require(f.get('runner_sha256') == base['runner_sha256'] == sha(runner), 'Extension runner differs from frozen base')
    require(f.get('model_id') == base['model_id'] and f.get('model_files') == base['model_files'],
            'Extension model provenance differs from base')
    expected_prompt, original_ids = base_long(base_root, base)
    require(set(f.get('files',{})) == set(FILES), 'Extension artifact set differs')
    for name in FILES:
        require(sha(directory/('decode32.'+name)) == f['files'][name], 'Extended decode artifact changed: '+name)
    prompt = (directory/'decode32.prompt.txt').read_text()
    native = json.loads((directory/'decode32.tokenization.json').read_text())
    ids = json.loads((directory/'decode32.tokens.json').read_text())
    payload = json.loads((directory/'decode32.request.json').read_text())
    require(prompt == expected_prompt, 'Extension must remove22 filler units and use the calibrated shorter tail')
    require(isinstance(ids,list) and len(ids) == 262112 and ids == native.get('tokens')
            and all(type(t) is int and 0 <= t < 2**31 for t in ids), 'Extension native token count/type differs')
    require(native.get('decoded') == native.get('rendered_prompt')
            and isinstance(native.get('rendered_prompt'),str) and native['rendered_prompt'].count(prompt.strip()) == 1,
            'Extension native roundtrip/render differs')
    require(payload == {'model':base['model_id'],'messages':[{'role':'user','content':prompt}],
                        'max_tokens':32,'stream':False,'mtp_depth':0}, 'Extension HTTP payload differs')
    lcp = next((i for i,(a,b) in enumerate(zip(original_ids,ids)) if a != b), min(len(original_ids),len(ids)))
    require(lcp >= 262080 and (len(ids)-1)//416*416 == 262080, 'Extension cannot reuse the original checkpoint')
    require(f.get('longest_common_prefix_tokens') == lcp and f.get('prompt_tokens') == 262112
            and f.get('max_tokens') == 32 and f.get('expected_cached_tokens') == 262080
            and f.get('expected_computed_tokens') == 32 and f.get('expected_decoded_tokens') == 31
            and f.get('expected_final_state_offset') == 262143, 'Extended expected counts differ from raw evidence')
    return {'payload':payload,'evidence':{'fixture_directory':str(directory.resolve()),
            'fixture_sha256':sha(directory/'fixture.json'),'base_fixture_sha256':f['base_fixture_sha256'],
            'longest_common_prefix_tokens':lcp,'prompt_tokens':len(ids),'max_tokens':32,
            'expected_cached_tokens':262080,'expected_computed_tokens':32,'expected_decoded_tokens':31,
            'expected_final_state_offset':262143,'independent_cold_oracle':False,'http_token_ids_available':False,
            'comparison_scope':'Exact complete response text, finish and usage; not output token IDs or quality.'}}


def prepare(args):
    runner, model, base_root = args.runner.resolve(), args.model_dir.resolve(), args.base_fixture.resolve()
    base = json.loads((base_root/'fixture.json').read_text())
    require(sha(runner) == base['runner_sha256'], 'Runner differs from frozen base fixture')
    require(model.name == base['model_id'], 'Model identifier differs from base')
    require(set(base['model_files']) == {'config.json','tokenizer.json','chat_template.jinja'}, 'Base model provenance missing')
    for name, expected in base['model_files'].items():
        require(sha(model/name) == expected, 'Model metadata differs: '+name)
    prompt, original_ids = base_long(base_root,base)
    out = args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    manifest = {'schema':'qwen-long-http-decode-fixture-v1','complete':False,
                'base_fixture_directory':str(base_root),'base_fixture_sha256':sha(base_root/'fixture.json'),
                'runner_sha256':sha(runner),'model_id':base['model_id'],'model_files':base['model_files'],
                'script_sha256':sha(__file__),'files':{},'independent_cold_oracle':False,
                'quality_evaluation':False,'gpu_executed':False,'cpu_tokenize_calls':1,
                'prompt_transform':{'removed_repeat_units':22,'new_tail':DECODE_TAIL,'tail_token_reduction':8}}
    try:
        (out/'decode32.prompt.txt').write_text(prompt,encoding='utf-8')
        command = [str(runner),'tokenize','--model-dir',str(model),'--prompt-file',str(out/'decode32.prompt.txt'),
                   '--chat','true','--output',str(out/'decode32.tokenization.json')]
        manifest['command'] = command
        with (out/'decode32.tokenizer.log').open('xb') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        manifest['tokenizer_exit_code'] = result.returncode
        require(result.returncode == 0, 'Native CPU tokenization failed')
        native = json.loads((out/'decode32.tokenization.json').read_text()); ids = native.get('tokens')
        require(isinstance(ids,list) and len(ids) == 262112 and all(type(t) is int and 0 <= t < 2**31 for t in ids),
                'Exact P262112 native token count failed')
        require(native.get('decoded') == native.get('rendered_prompt') and isinstance(native.get('rendered_prompt'),str)
                and native['rendered_prompt'].count(prompt.strip()) == 1, 'Native roundtrip/render failed')
        lcp = next((i for i,(a,b) in enumerate(zip(original_ids,ids)) if a != b), min(len(original_ids),len(ids)))
        require(lcp >= 262080 and (len(ids)-1)//416*416 == 262080, 'Checkpoint reuse failed')
        dump(out/'decode32.tokens.json',ids)
        payload = {'model':model.name,'messages':[{'role':'user','content':prompt}],'max_tokens':32,'stream':False,'mtp_depth':0}
        dump(out/'decode32.request.json',payload)
        manifest.update(complete=True,prompt_tokens=len(ids),max_tokens=32,longest_common_prefix_tokens=lcp,
                        expected_cached_tokens=262080,expected_computed_tokens=32,expected_decoded_tokens=31,
                        expected_final_state_offset=262143,body_bytes=len(json.dumps(payload,ensure_ascii=False).encode()),
                        files={name:sha(out/('decode32.'+name)) for name in FILES})
    except BaseException as error:
        manifest['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        dump(out/'fixture.json',manifest)
    load_decode_fixture(out,base_root,base,runner)
    print(json.dumps({'event':'decode_fixture_prepared','output':str(out),'prompt_tokens':262112,
                      'longest_common_prefix_tokens':lcp,'expected_cached_tokens':262080,'gpu_executed':False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runner','model-dir','base-fixture','output'):
        p.add_argument('--'+name,type=Path,required=True)
    prepare(p.parse_args())
