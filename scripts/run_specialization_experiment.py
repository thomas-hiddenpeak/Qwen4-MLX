"""Run a saved experiment plan with one model owner and finally-based restoration.

Plan commands are local subprocess argument arrays. Token and capture gates stop
the plan before subsequent benchmarks; all process identities/results are saved.
"""
from pathlib import Path
import datetime, hashlib, json, os, signal, struct, subprocess, sys, time, urllib.request
from owned_process_group import OwnedProcessGroup

ROOT = Path(__file__).resolve().parents[1]
plan_path = Path(sys.argv[1]).resolve()
OUT = plan_path.parent
plan = json.loads(plan_path.read_text())
# The nested MTP analysis plan is deliberately not an execution plan. Reject
# it before pausing the reference or creating an experiment ledger.
assert isinstance(plan.get('cases'), list) and plan['cases'] and all(
    isinstance(case.get('name'), str) and isinstance(case.get('command'), list)
    and case['command'] for case in plan['cases']), 'Expected flat executable cases with name/command'
STATUS = ROOT / '../qwen38-ssd/results/experiment-status.json'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(): (OUT / 'run-ledger.json').write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n')
def identity(pid): return subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
def http(path):
    with urllib.request.urlopen('http://127.0.0.1:11235' + path, timeout=3) as response: return response.read().decode()
stop_signal = None
def request_stop(number, _frame):
    global stop_signal
    # Never throw between fork/exec and recording the newly owned Popen/group.
    if stop_signal is None: stop_signal = number
def check_stop():
    if stop_signal is not None: raise InterruptedError(f'Controller received signal {stop_signal}')
def wait_case(owned, timeout):
    deadline = time.monotonic() + timeout
    while True:
        check_stop()
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise subprocess.TimeoutExpired(owned.process.args, timeout)
        try: return owned.process.wait(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            if time.monotonic() >= deadline: raise
def reference_ready():
    meta = json.loads(http('/v1/models'))['data'][0]['meta']
    return meta.get('mtp_loaded') is False and meta.get('drafter_loaded') is False
def tensors(path):
    data = Path(path).read_bytes()
    length = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8+length])
    payload = memoryview(data)[8+length:]
    return {k: (v['dtype'], v['shape'], bytes(payload[slice(*v['data_offsets'])]))
            for k, v in header.items() if k != '__metadata__'}

reference = json.loads(Path(plan['reference_ledger']).read_text())['restoration']
status = json.loads(STATUS.read_text())
pid, argv = reference['pid'], reference['argv']
assert status['server_pid'] == pid and identity(pid) == ' '.join(argv), 'Reference identity mismatch'
metrics = http('/metrics')
for key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
    values = [float(line.split()[-1]) for line in metrics.splitlines() if line.startswith(key + ' ')]
    assert values and all(v == 0 for v in values), 'Reference is not verifiably idle'
assert all(flag in argv for flag in ('--no-mtp', '--no-drafter', '--no-pld'))
assert not (OUT / 'run-ledger.json').exists(), 'Refuse to overwrite an experiment'
binary = ROOT / '.build/release/ane-runner'
ledger = {'started_utc': now(), 'original_reference_pid': pid, 'reference_argv': argv,
          'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'runs': [], 'restoration': None}
save()
owned = None
run = None
env = os.environ.copy()
env['DEVELOPER_DIR'] = '/Applications/Xcode.app/Contents/Developer'
for key in ('MLX_MAX_MB_PER_BUFFER', 'MLX_MAX_OPS_PER_BUFFER'): env.pop(key, None)
env.update(plan.get('environment', {}))
print('Pausing verified idle reference', flush=True)
signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)
try:
    check_stop()
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 30
    while identity(pid) and time.monotonic() < deadline: time.sleep(0.1)
    assert not identity(pid), 'Reference did not stop'
    status.update(phase='paused_for_specialization_validation', server_retained=False, updated_utc=now())
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    check_stop()
    for case in plan['cases']:
        check_stop()
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == ledger['binary_sha256'], 'Binary changed during experiment'
        run = {'name': case['name'], 'command': case['command'], 'started_utc': now()}
        ledger['runs'].append(run); save()
        print('Starting ' + case['name'], flush=True)
        with (OUT / (case['name'] + '.log')).open('xb') as log:
            owned = OwnedProcessGroup.start(case['command'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            run.update(pid=owned.pid, pgid=owned.pgid); save()
            try: wait_case(owned, case.get('timeout', 600))
            except subprocess.TimeoutExpired: run['timeout'] = True
            cleanup = owned.finish()  # 45s TERM grace, then at most 10s after KILL.
            run['cleanup'] = cleanup.report()
            run.update(exit_code=owned.process.returncode, ended_utc=now()); save()
            if not cleanup.cleaned:
                raise RuntimeError('Owned experiment group did not exit; reference restoration is blocked')
            owned = None  # Never reuse a pgid after observing that its group disappeared.
        check_stop()
        assert not run.get('timeout'), case['name'] + ' timed out; see log'
        assert run['exit_code'] == 0 and not cleanup.kill_sent, case['name'] + ' failed; see log'
        if 'capture_pair' in case:
            lhs, rhs = (tensors(p) for p in case['capture_pair'])
            mismatches = [k for k in lhs.keys() | rhs.keys() if lhs.get(k) != rhs.get(k)]
            run.update(capture_tensor_count=len(lhs), capture_bitwise_equal=not mismatches, mismatches=mismatches)
            save(); print('Capture bitwise gate: ' + str(not mismatches), flush=True)
            assert not mismatches, 'Real tensor gate failed'
        if 'generation_report' in case:
            report = json.loads(Path(case['generation_report']).read_text())
            golden = json.loads(Path(case['golden_report']).read_text())['trials'][0]['generated_token_ids']
            trials = report['trials']
            equal = all(t['generated_token_ids'] == golden for t in trials)
            run.update(exact_token_match=equal, trial_modes=[t['decode_mode'] for t in trials],
                       trial_tps=[t['decode_tokens_per_second'] for t in trials],
                       trial_wired=[t['wired_memory']['policy'] for t in trials])
            save(); print(json.dumps({k: run[k] for k in ('name', 'exact_token_match', 'trial_modes', 'trial_tps', 'trial_wired')}), flush=True)
            assert equal, 'Full model token gate failed'
except BaseException as error:
    ledger['error'] = f'{type(error).__name__}: {error}'
    if stop_signal is not None: ledger['interrupted_signal'] = stop_signal
    raise
finally:
    # The flag-only handlers stay installed. Repeated TERM/INT cannot interrupt
    # cleanup or strand a half-created restoration process.
    cleanup = owned.finish() if owned is not None else None
    if cleanup is not None:
        run.update(cleanup=cleanup.report(), exit_code=owned.process.returncode, ended_utc=now())
    if stop_signal is not None: ledger['interrupted_signal'] = stop_signal
    if cleanup is not None and not cleanup.cleaned:
        ledger['restoration'] = {'ready': False, 'blocked_by_owned_group': cleanup.report()}
        status.update(phase='reference_restore_blocked_by_owned_group', server_retained=False, updated_utc=now())
        save()
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        raise RuntimeError('Cannot prove experiment group exited; refusing to load another model')
    owned = None
    current_reference = identity(pid)
    if current_reference:
        restoration = dict(reference, ready=False, retained_original=True)
        ledger['restoration'] = restoration
        if current_reference != ' '.join(argv):
            restoration['error'] = 'Original reference PID no longer matches its verified argv'
        else:
            try:
                if reference_ready() and identity(pid) == ' '.join(argv):
                    restoration.update(ready=True, ready_verified_utc=now(), mtp_loaded=False, drafter_loaded=False)
                else: restoration['error'] = 'Retained reference is not ready with MTP/drafter disabled'
            except Exception as error:
                restoration['error'] = f'Retained reference readiness failed: {error}'
        status.update(phase='ready_reference_retained_after_specialization' if restoration['ready'] else 'reference_restore_needs_attention',
                      server_retained=restoration['ready'], reference_ledger=str(OUT / 'run-ledger.json'), updated_utc=now())
        save()
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        assert restoration['ready'], 'Could not verify retained reference; no new model was started'
    else:
        print('Restoring reference with MTP disabled', flush=True)
        with (OUT / 'reference-restored.log').open('xb') as log:
            server = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        restoration = {'pid': server.pid, 'argv': argv, 'log': str(OUT / 'reference-restored.log'), 'started_utc': now(), 'ready': False}
        ledger['restoration'] = restoration; save()
        status.update(server_pid=server.pid, server_log=restoration['log'], server_retained=True, phase='restoring_reference_after_specialization', updated_utc=now())
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and server.poll() is None:
            try:
                if reference_ready():
                    restoration.update(ready=True, ready_verified_utc=now(), mtp_loaded=False, drafter_loaded=False); break
            except Exception: pass
            time.sleep(1)
        if stop_signal is not None: ledger['interrupted_signal'] = stop_signal
        save()
        status.update(phase='ready_reference_restored_after_specialization' if restoration['ready'] else 'reference_restore_needs_attention',
                      reference_ledger=str(OUT / 'run-ledger.json'), updated_utc=now())
        STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
        assert restoration['ready'], 'Could not verify restored reference'
        print('Reference ready: PID ' + str(server.pid), flush=True)
