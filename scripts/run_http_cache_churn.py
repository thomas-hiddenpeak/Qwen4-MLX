"""Own one server within the experiment controller's existing process group."""
from pathlib import Path
import argparse, http.client, json, os, signal, socket, subprocess, sys, threading, time

parser = argparse.ArgumentParser()
parser.add_argument('--output-directory', type=Path, required=True)
parser.add_argument('--duration-seconds', type=int, default=7200)
parser.add_argument('--port', type=int, default=11248)
parser.add_argument('--fixture-mode', choices=('legacy', 'counter-witness'), default='legacy')
parser.add_argument('--kv-append-mode', choices=('reference', 'capacity256'), default='reference')
parser.add_argument('--paged-kv-pool-library', type=Path)
parser.add_argument('--paged-kv-pages-per-layer', type=int)
parser.add_argument('--prefix-cache-bytes', type=int, default=167772160)
parser.add_argument('--state-budget-bytes', type=int, default=4294967296)
args = parser.parse_args()
if not 1 <= args.duration_seconds <= 7200:
    parser.error('--duration-seconds must be in 1...7200')
if not 1024 <= args.port <= 65535:
    parser.error('--port must be in 1024...65535')
if not 1 <= args.prefix_cache_bytes <= 8589934592 or args.state_budget_bytes < 1:
    parser.error('Explicit RAM cache/state budget must be positive and RAM cache <=8GiB')
if (args.paged_kv_pool_library is None) != (args.paged_kv_pages_per_layer is None):
    parser.error('--paged-kv-pool-library and --paged-kv-pages-per-layer must be supplied together')
if args.paged_kv_pool_library is not None:
    if (not args.paged_kv_pool_library.is_absolute() or not args.paged_kv_pool_library.is_file()
            or not os.access(args.paged_kv_pool_library, os.R_OK) or not 1 <= args.paged_kv_pages_per_layer <= 4096):
        parser.error('Paged pool requires a readable absolute library and pages 1...4096')
    if args.kv_append_mode != 'reference': parser.error('Paged pool cannot use capacity256')
    if args.prefix_cache_bytes < 805306368 or args.state_budget_bytes < 8589934592:
        parser.error('Paged churn requires explicit --prefix-cache-bytes >=805306368 and --state-budget-bytes >=8589934592')
root = Path.cwd(); out = args.output_directory.resolve()
sys.path.insert(0, str(root / 'scripts'))
from probe_http_cache_reliability import health_contract, fully_idle
from capacity_http_validation import require_capacity_health, validate_capacity_terminals
from paged_http_validation import require_paged_health, validate_paged_terminals

port = args.port; child = None; churn = None; sample_thread = None
sample_stop = threading.Event(); interrupted = threading.Event(); sample_failed = threading.Event()
sampling = {'completed': False, 'samples': 0, 'rss_samples': 0, 'fd_samples': 0,
            'errors': 0, 'error': None, 'max_samples': 1024, 'interval_seconds': 30}
record = {'complete': False, 'passed': False, 'services': [],
          'fixture_mode': args.fixture_mode, 'kv_append_mode': args.kv_append_mode,
          'paged_kv_pool_library': str(args.paged_kv_pool_library) if args.paged_kv_pool_library else None,
          'paged_kv_pages_per_layer': args.paged_kv_pages_per_layer,
          'prefix_cache_bytes': args.prefix_cache_bytes, 'state_budget_bytes': args.state_budget_bytes,
          'sampling': sampling, 'interrupted_by_signal': None,
          'process_observations': 'RSS and numeric FDs, not unique physical Metal memory or device I/O'}
def save():
    temporary = out / 'lifecycle.json.tmp'
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(out / 'lifecycle.json')

def get(path):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
    try:
        connection.request('GET', path)
        response = connection.getresponse(); data = response.read(4 * 1024 * 1024 + 1)
        if response.status != 200 or len(data) > 4 * 1024 * 1024:
            raise ValueError(f'Unexpected HTTP response status/size: {response.status}/{len(data)}')
        return data
    finally:
        connection.close()

def idle():
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        check_stop()
        if child.poll() is not None:
            raise RuntimeError('Server exited before idle')
        try:
            value = health_contract(json.loads(get('/health')))
            if args.kv_append_mode == 'capacity256': require_capacity_health(value)
            if args.paged_kv_pool_library: require_paged_health(value, args.paged_kv_pages_per_layer)
            if value['pid'] != child.pid:
                raise RuntimeError('Health PID does not match the owned server')
            if fully_idle(value): return value
        except (OSError, ValueError): pass
        time.sleep(.1)
    raise TimeoutError('Server did not drain within 180 seconds')

def sample_process(proc):
    started = time.monotonic()
    try:
        with (out / 'process.ndjson').open('x') as stream:
            while not sample_stop.is_set() and proc.poll() is None:
                if sampling['samples'] >= sampling['max_samples']:
                    raise RuntimeError('Process sample bound exhausted')
                value = {'elapsed_seconds': time.monotonic() - started,
                         'monotonic_seconds': time.monotonic(), 'pid': proc.pid}
                try:
                    observed = subprocess.run(['ps', '-p', str(proc.pid), '-o', 'rss='],
                                              capture_output=True, text=True, timeout=3)
                    if observed.returncode != 0 or not observed.stdout.strip():
                        raise RuntimeError('RSS observation failed')
                    value['rss_bytes'] = int(observed.stdout.strip()) * 1024
                    files = subprocess.run(['/usr/sbin/lsof', '-nP', '-p', str(proc.pid)],
                                           capture_output=True, text=True, timeout=3)
                    if files.returncode != 0:
                        raise RuntimeError('FD observation failed')
                    value['numeric_fds'] = sum(len(parts) > 3 and parts[3].rstrip('ruw').isdigit()
                        for parts in (line.split() for line in files.stdout.splitlines()[1:]))
                    if value['rss_bytes'] <= 0 or value['numeric_fds'] <= 0:
                        raise RuntimeError('Empty RSS/FD observation')
                    health = health_contract(json.loads(get('/health')))
                    if health['pid'] != proc.pid:
                        raise RuntimeError('Process sample health PID mismatch')
                    value.update(idle=health['idle'], fully_idle=fully_idle(health),
                                 state_budget=health['state_budget'], mlx_memory=health['mlx_memory'],
                                 memory_pressure=health['memory_pressure'],
                                 memory_pressure_monitor_running=health['memory_pressure_monitor_running'],
                                 prefix_cache=health['prefix_cache'], prefix_disk_cache=health['prefix_disk_cache'])
                    value['paged_kv_pool'] = health.get('paged_kv_pool')
                except Exception:
                    # A sample already in progress may lose its process when
                    # cleanup starts. This is not a workload observation.
                    if sample_stop.is_set(): break
                    raise
                stream.write(json.dumps(value, separators=(',', ':')) + '\n'); stream.flush()
                sampling['samples'] += 1; sampling['rss_samples'] += 1; sampling['fd_samples'] += 1
                sample_stop.wait(sampling['interval_seconds'])
        if not sample_stop.is_set():
            raise RuntimeError('Server exited while process sampling was active')
        sampling['completed'] = True
    except BaseException as error:
        sampling['errors'] += 1
        sampling['error'] = f'{type(error).__name__}: {error}'[:500]
        sample_failed.set()

def check_stop():
    if interrupted.is_set(): raise InterruptedError('Wrapper received a stop signal')
    if sample_failed.is_set(): raise RuntimeError('Process sampling failed: ' + sampling['error'])

def on_signal(number, _frame):
    record['interrupted_by_signal'] = number
    interrupted.set(); sample_stop.set()

def check_working_set(result):
    if 'working_set' not in result: return False
    measured = result['working_set']
    assert measured['ssd_limit_bytes'] == 1073741824
    ratio = measured['measured_distinct_published_archive_bytes'] / measured['ssd_limit_bytes']
    record['measured_working_set_to_ssd_ratio'] = ratio
    record['working_set_validated_at_soak_start'] = 2 <= ratio <= 4
    save()
    assert 2 <= ratio <= 4, ('Working set outside release candidate target', ratio)
    return True

# Take exclusive ownership before entering the cleanup block. A rejected repeat
# invocation must not overwrite another run's lifecycle report in finally.
out.mkdir(parents=True, exist_ok=True)
with (out / 'lifecycle.json').open('x') as stream:
    json.dump(record, stream, indent=2); stream.write('\n')
for number in (signal.SIGINT, signal.SIGTERM): signal.signal(number, on_signal)
workload_passed = False

try:
    with socket.socket() as available: available.bind(('127.0.0.1', port))
    model = root.parent / 'qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream'
    command = [str(root / '.build/release/ane-runner'), 'serve-gpu', '--model-dir', str(model),
        '--port', str(port), '--prefix-cache-directory', str(out / 'cache'),
        '--prefix-cache-bytes', str(args.prefix_cache_bytes), '--prefix-cache-disk-bytes', '1073741824',
        '--prefix-cache-entries', '8', '--prefix-cache-disk-entries', '16',
        '--prefix-cache-ttl-seconds', '86400', '--prefix-cache-min-free-bytes', '1073741824',
        '--prefix-cache-restore-timeout-seconds', '5', '--prefix-cache-shutdown-timeout-seconds', '30',
        '--state-budget-bytes', str(args.state_budget_bytes), '--max-connections', '8']
    if args.kv_append_mode == 'capacity256': command.extend(['--kv-append-mode', 'capacity256'])
    if args.paged_kv_pool_library:
        command.extend(['--paged-kv-pool-library', str(args.paged_kv_pool_library.resolve()),
                        '--paged-kv-pages-per-layer', str(args.paged_kv_pages_per_layer)])
    with (out / 'server.log').open('x') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    row = {'pid': child.pid, 'command': command}; record['services'].append(row); save()
    row['initial_health'] = idle()
    assert row['initial_health']['prefix_cache_policy']['lookup'] == 'complete_canonical_prompt'
    (out / 'initial.prom').write_bytes(get('/metrics'))
    sample_thread = threading.Thread(target=sample_process, args=(child,), daemon=True)
    sample_thread.start(); save()
    churn_command = [sys.executable, '-B', str(root / 'scripts/probe_http_cache_churn.py'),
        '--base-url', f'http://127.0.0.1:{port}', '--duration-seconds', str(args.duration_seconds),
        '--seed', '20260909', '--concurrency', '4', '--long-prefixes', '8', '--short-prefixes', '2',
        '--max-inflight-prompt-tokens', '30000', '--request-timeout-seconds', '240',
        '--drain-timeout-seconds', '180', '--drain-interval-seconds', '300', '--health-interval-seconds', '2',
        '--fixture-mode', args.fixture_mode,
        '--output', str(out / 'churn.json')]
    churn = subprocess.Popen(churn_command, stdin=subprocess.DEVNULL)
    record['client'] = {'pid': churn.pid, 'command': churn_command}; save()
    workset_checked = False
    while churn.poll() is None:
        check_stop()
        if child.poll() is not None: raise RuntimeError('Server exited during churn')
        if not workset_checked and (out / 'churn.json').exists():
            workset_checked = check_working_set(json.loads((out / 'churn.json').read_text()))
        interrupted.wait(1)
    check_stop()
    record['client']['exit_code'] = churn.returncode
    assert churn.returncode == 0, f'Churn exited {churn.returncode}'
    row['final_health'] = idle()
    (out / 'final.prom').write_bytes(get('/metrics'))
    result = json.loads((out / 'churn.json').read_text())
    assert result['passed'] and result['complete']
    assert check_working_set(result)
    assert row['final_health']['memory_pressure_monitor_running'] is True
    workload_passed = True
except BaseException as error:
    record['error'] = f'{type(error).__name__}: {error}'[:1500]
finally:
    # Stop observations and signal both owned children immediately. Their waits
    # and the sampler join share one deadline below the controller's 45s grace.
    sample_stop.set()
    shutdown_deadline = time.monotonic() + 40
    for proc in (child, churn):
        if proc is not None and proc.poll() is None:
            try: proc.terminate()
            except ProcessLookupError: pass
    cleanup = {}
    for name, proc in (('server', child), ('client', churn)):
        if proc is None: continue
        try: cleanup[name + '_exit_code'] = proc.wait(timeout=max(0, shutdown_deadline - time.monotonic()))
        except subprocess.TimeoutExpired: cleanup[name + '_pending_pid'] = proc.pid
    if sample_thread is not None:
        sample_thread.join(timeout=max(0, shutdown_deadline - time.monotonic()))
        cleanup['sampler_completed'] = not sample_thread.is_alive() and sampling['completed']
    record['cleanup'] = cleanup
    if child is not None:
        row['exit_code'] = cleanup.get('server_exit_code')
        row['close_completed_logged'] = 'HTTP prefix cache shutdown completed=true io_completed=true callbacks_completed=true pending_jobs=0 pending_bytes=0' in (out / 'server.log').read_text()
    capacity_passed = args.kv_append_mode == 'reference'
    record['capacity_terminal_validation'] = {'complete': False, 'passed': False, 'skipped': True,
                                             'required': not capacity_passed}
    if args.kv_append_mode == 'capacity256' and workload_passed and not interrupted.is_set() and cleanup.get('server_exit_code') == 0:
        try:
            audit = validate_capacity_terminals(out / 'churn.events.ndjson', out / 'server.log', child.pid,
                                               stop_requested=interrupted.is_set)
            record['capacity_terminal_validation'] = audit
            with (out / 'capacity-terminal-validation.json').open('x') as stream:
                json.dump(audit, stream, indent=2); stream.write('\n')
            capacity_passed = audit['complete'] and audit['passed']
        except (OSError, ValueError, TypeError) as error:
            record['capacity_terminal_validation'] = {'complete': False, 'passed': False,
                'errors': [f'{type(error).__name__}: {error}'[:1500]]}
    paged_passed = args.paged_kv_pool_library is None
    record['paged_terminal_validation'] = {'complete': False, 'passed': False, 'skipped': True, 'required': not paged_passed}
    if not paged_passed and workload_passed and not interrupted.is_set() and cleanup.get('server_exit_code') == 0:
        try:
            audit = validate_paged_terminals(out / 'churn.events.ndjson', out / 'server.log', child.pid,
                row['initial_health'], row['final_health'], stop_requested=interrupted.is_set)
            record['paged_terminal_validation'] = audit
            with (out / 'paged-terminal-validation.json').open('x') as stream:
                json.dump(audit, stream, indent=2); stream.write('\n')
            paged_passed = audit['complete'] and audit['passed']
        except (OSError, ValueError, TypeError) as error:
            record['paged_terminal_validation'] = {'complete': False, 'passed': False,
                'errors': [f'{type(error).__name__}: {error}'[:1500]]}
    record['complete'] = record['passed'] = bool(workload_passed and not interrupted.is_set()
        and cleanup.get('server_exit_code') == 0 and cleanup.get('client_exit_code') == 0
        and cleanup.get('sampler_completed') and sampling['samples'] > 0 and sampling['errors'] == 0
        and row['close_completed_logged'] and capacity_passed and paged_passed)
    if workload_passed and not record['passed']:
        record['error'] = 'Post-workload sampler/close/exit verification failed; inspect cleanup and service records'
    save()
raise SystemExit(0 if record['passed'] else 1)
