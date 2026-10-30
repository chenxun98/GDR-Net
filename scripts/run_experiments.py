#!/usr/bin/env python3
"""Run, inspect, or stop one experiment manifest without broad process matching.

Run from the project root. Training and external evaluation are separate manifest
stages; external evaluation reuses completed source checkpoints and thresholds.
Real launches and process-group stopping target Linux GPU servers. Inspection,
dry runs and progress checks also work on Windows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def nonempty_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def source_complete(run):
    """A process exit or best.pt alone does not establish completed training."""
    run = Path(run)
    names = ('training_complete.json', 'history.csv', 'best.pt',
             'ensemble_checkpoints.json', 'ensemble_val_metrics.json', 'used_config.yaml')
    try:
        if not all(nonempty_file(run / name) for name in names):
            return False
        epochs = int(read_json(run / 'training_complete.json')['epochs_completed'])
        with (run / 'history.csv').open(encoding='utf-8', newline='') as handle:
            history = list(csv.DictReader(handle))
        if epochs < 1 or len(history) != epochs:
            return False
        if not isinstance(read_json(run / 'ensemble_val_metrics.json'), dict):
            return False
        if not isinstance(load_config(run / 'used_config.yaml'), dict):
            return False
        paths = [Path(item['path']) for item in read_json(run / 'ensemble_checkpoints.json')]
        # train.py stores paths relative to the launch directory (project root).
        return len({str(path.resolve()) for path in paths}) >= 2 and all(nonempty_file(path) for path in paths)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        return False


def external_complete(run):
    path = Path(run) / 'external_evaluation' / 'ensemble_result_manifest.json'
    try:
        datasets = read_json(path)['datasets']
        if not isinstance(datasets, dict) or not datasets:
            return False
        for files in datasets.values():
            for required in ('metrics', 'window_predictions', 'events', 'reliability'):
                if not files.get(required) or not nonempty_file(path.parent / files[required]):
                    return False
            if files.get('type_confusion_matrix') and not nonempty_file(path.parent / files['type_confusion_matrix']):
                return False
            with (path.parent / files['metrics']).open(encoding='utf-8', newline='') as handle:
                if not next(csv.DictReader(handle), None):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_manifest(path):
    payload = read_json(path)
    if payload.get('stage') not in ('source', 'external'):
        raise ValueError('Manifest stage must be source or external.')
    items = payload.get('experiments')
    if not isinstance(items, list) or not items:
        raise ValueError('Manifest experiments must be a nonempty list.')
    ids, outputs = set(), set()
    for item in items:
        for key in ('id', 'task', 'model', 'fold', 'seed', 'config', 'output_dir'):
            if key not in item:
                raise ValueError(f'Missing experiment field: {key}')
        identifier = str(item['id'])
        output = str(Path(item['output_dir']).resolve())
        if identifier in ids or output in outputs:
            raise ValueError('Duplicate experiment id or output directory in manifest.')
        if item.get('server') not in (3, 6):
            raise ValueError(f'{identifier}: server must be 3 or 6.')
        ids.add(identifier)
        outputs.add(output)
        key = 'train_argv' if payload['stage'] == 'source' else 'external_argv'
        argv = item.get(key)
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise ValueError(f'{identifier}: {key} must contain script name and arguments.')
        script = 'train.py' if payload['stage'] == 'source' else 'evaluate_external.py'
        if Path(argv[0]).name != script:
            raise ValueError(f'{identifier}: expected {script} as first argument.')
        if '--overwrite' in argv:
            raise ValueError(f'{identifier}: use the launcher --restart-incomplete flag instead of embedding --overwrite.')
        required_paths = {'--config': item['config'], '--output_dir': item['output_dir']} if payload['stage'] == 'source' else {'--run_dir': item['output_dir']}
        for option, expected in required_paths.items():
            if argv.count(option) != 1 or argv.index(option) + 1 >= len(argv):
                raise ValueError(f'{identifier}: command must provide {option} exactly once.')
            actual = argv[argv.index(option) + 1]
            if Path(actual).resolve() != Path(expected).resolve():
                raise ValueError(f'{identifier}: {option} differs from the manifest metadata.')
    return payload


def build_command(item, stage, restart_incomplete=False):
    """Return None for complete work; refuse mismatched or partial output by default."""
    run = Path(item['output_dir'])
    expected = load_config(item['config'])
    if (run / 'used_config.yaml').exists() and load_config(run / 'used_config.yaml') != expected:
        raise ValueError('Existing run configuration differs; use a new output directory.')
    if stage == 'external':
        if not source_complete(run):
            raise ValueError('Source training and ensemble artifacts are incomplete.')
    complete = source_complete if stage == 'source' else external_complete
    if complete(run):
        return None
    argv = [sys.executable, *item['train_argv' if stage == 'source' else 'external_argv']]
    existing = run if stage == 'source' else run / 'external_evaluation'
    if existing.exists() and any(existing.iterdir()):
        if not restart_incomplete:
            raise ValueError('Incomplete output exists. Use --restart-incomplete for an intentional restart.')
        if '--overwrite' not in argv:
            argv.append('--overwrite')
    return argv


def runtime_directory(manifest):
    key = hashlib.sha256(str(Path(manifest).resolve()).encode()).hexdigest()[:12]
    return Path('runs/launcher') / (Path(manifest).stem + '_' + key)


class ManifestLock:
    """OS lock released on process death; a leftover lock file is harmless."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        try:
            if os.name == 'nt':
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.handle.write(b'0')
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError('This manifest is already running.') from exc
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            if os.name == 'nt':
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def process_identity(pid):
    """PID plus kernel boot id and process start ticks defeats PID reuse."""
    if not sys.platform.startswith('linux'):
        return None
    try:
        directory = Path('/proc') / str(int(pid))
        # comm can contain spaces and parentheses; split after its last ')'.
        fields = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': int(fields[19]),
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'pgid': int(fields[2]), 'session': int(fields[3]),
                'uid': directory.stat().st_uid}
    except (OSError, ValueError, IndexError):
        return None


def same_process(record):
    if not isinstance(record, dict) or 'pid' not in record:
        return False
    current = process_identity(record['pid'])
    return current is not None and all(current.get(k) == record.get(k) for k in ('pid', 'start_ticks', 'boot_id', 'uid'))


def group_members(leader):
    """Remember live descendants so an exited leader need not orphan workers."""
    if not leader or not sys.platform.startswith('linux'):
        return []
    members = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        current = process_identity(int(path.name))
        if (current and current['pgid'] == leader['pgid'] and current['session'] == leader['session']
                and current['uid'] == leader['uid'] and current['start_ticks'] >= leader['start_ticks']):
            members.append(current)
    return members


def group_is_owned(job):
    leader = job.get('process') or {}
    pgid = leader.get('pgid')
    if not pgid or pgid <= 1 or leader.get('session') != pgid or leader.get('pid') != pgid:
        return False
    for recorded in [leader, *job.get('members', [])]:
        if same_process(recorded):
            current = process_identity(recorded['pid'])
            if current and current['pgid'] == pgid and current['session'] == pgid:
                return True
    return False


def signal_group(job, sig):
    if not group_is_owned(job):
        return False
    try:
        os.killpg(job['process']['pgid'], sig)
        return True
    except ProcessLookupError:
        return False


def reap_remaining_group(job, grace_seconds=3):
    """A stopped or failed training leader must not leave registered workers."""
    if not signal_group(job, signal.SIGTERM):
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline and group_is_owned(job):
        time.sleep(.1)
    signal_group(job, signal.SIGKILL)


def latest_epoch(run, log=None):
    try:
        with (Path(run) / 'history.csv').open(encoding='utf-8', newline='') as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return int(float(rows[-1]['epoch']))
    except (OSError, ValueError, KeyError):
        pass
    if log:
        try:
            with Path(log).open('rb') as handle:
                handle.seek(max(0, handle.seek(0, 2) - 262144))
                lines = handle.read().decode('utf-8', errors='replace').splitlines()
            for line in reversed(lines):
                if line.strip().startswith('{'):
                    try:
                        value = json.loads(line)
                        if 'epoch' in value:
                            return int(value['epoch'])
                    except (ValueError, TypeError):
                        continue
        except OSError:
            pass
    return None


def read_state(manifest):
    path = runtime_directory(manifest) / 'state.json'
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def status(manifest):
    payload = load_manifest(manifest)
    state = read_state(manifest)
    complete = source_complete if payload['stage'] == 'source' else external_complete
    rows = []
    for item in payload['experiments']:
        previous = state.get('jobs', {}).get(str(item['id']), {})
        run = Path(item['output_dir'])
        existing = run if payload['stage'] == 'source' else run / 'external_evaluation'
        if complete(run):
            label = 'complete'
        elif group_is_owned(previous):
            label = 'running'
        elif previous.get('status') == 'failed':
            label = 'failed'
        elif existing.exists() and any(existing.iterdir()):
            label = 'incomplete'
        else:
            label = 'not_started'
        row = {'id': item['id'], 'status': label, 'epoch': latest_epoch(run, previous.get('log'))}
        if previous.get('error'):
            row['error'] = previous['error']
        rows.append(row)
    counts = Counter(row['status'] for row in rows)
    result = {'stage': payload['stage'], 'total': len(rows), 'remaining': len(rows) - counts['complete'],
              'counts': {key: counts[key] for key in ('complete', 'running', 'not_started', 'failed', 'incomplete')},
              'experiments': rows}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(manifest, gpus, restart_incomplete=False, dry_run=False):
    payload = load_manifest(manifest)
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError('GPU identifiers must be nonempty and unique.')
    if dry_run:
        failures = 0
        for index, item in enumerate(payload['experiments']):
            try:
                argv = build_command(item, payload['stage'], restart_incomplete)
                print(json.dumps({'id': item['id'], 'gpu': str(gpus[index % len(gpus)]),
                                  'status': 'skip_complete' if argv is None else 'planned', 'argv': argv}))
            except Exception as exc:
                failures += 1
                print(json.dumps({'id': item['id'], 'error': str(exc)}))
        return int(failures > 0)
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Launch on the Linux GPU server; --dry-run and status also work locally.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --dry-run to inspect commands.')
    directory = runtime_directory(manifest)
    with ManifestLock(directory / 'run.lock'):
        old = read_state(manifest)
        if any(group_is_owned(job) for job in old.get('jobs', {}).values()):
            raise RuntimeError('Live experiment children remain. Stop this manifest before relaunching.')
        return _run_locked(manifest, payload, gpus, restart_incomplete, directory)


def _run_locked(manifest, payload, gpus, restart_incomplete, directory):
    stop_event = threading.Event()
    state_lock = threading.Lock()
    jobs = queue.Queue()
    for item in payload['experiments']:
        jobs.put(item)
    state = {'manifest': str(Path(manifest).resolve()), 'manifest_sha256': sha256(manifest),
             'stage': payload['stage'], 'launcher': process_identity(os.getpid()),
             'started_at': time.time(), 'status': 'running',
             'jobs': {str(item['id']): {'status': 'pending'} for item in payload['experiments']}}

    def save(identifier=None, **values):
        with state_lock:
            if identifier is not None:
                state['jobs'][identifier].update(values)
            state['updated_at'] = time.time()
            atomic_json(directory / 'state.json', state)

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            env.setdefault(key, '4')
        while not stop_event.is_set():
            try:
                item = jobs.get_nowait()
            except queue.Empty:
                return
            identifier = str(item['id'])
            process = None
            job = {}
            try:
                argv = build_command(item, payload['stage'], restart_incomplete)
                if argv is None:
                    save(identifier, status='complete')
                    print('SKIP', identifier, flush=True)
                    continue
                if stop_event.is_set():
                    break
                tag = hashlib.sha256(identifier.encode()).hexdigest()[:12]
                log_path = directory / (tag + '.log')
                print('START', identifier, 'GPU', gpu, 'log', log_path, flush=True)
                with log_path.open('a', encoding='utf-8') as log:
                    log.write('\n' + json.dumps({'launcher_started': time.time(), 'argv': argv}) + '\n')
                    log.flush()
                    process = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    leader = process_identity(process.pid)
                    if leader is None and process.poll() is None:
                        raise RuntimeError('Could not register child process identity.')
                    job = {'process': leader, 'members': [leader] if leader else []}
                    save(identifier, status='running', gpu=str(gpu), log=str(log_path), **job)
                    stopping_since = None
                    last_refresh = 0.0
                    while process.poll() is None:
                        now = time.monotonic()
                        if now - last_refresh > 2:
                            job['members'] = group_members(leader)
                            save(identifier, **job)
                            last_refresh = now
                        if stop_event.is_set():
                            if stopping_since is None:
                                signal_group(job, signal.SIGTERM)
                                stopping_since = now
                            elif now - stopping_since > 10:
                                signal_group(job, signal.SIGKILL)
                        time.sleep(.2)
                    code = process.wait()
                    # DataLoader workers may briefly outlive any training leader.
                    reap_remaining_group(job)
                if stop_event.is_set():
                    save(identifier, status='interrupted', exit_code=code)
                elif code != 0:
                    raise RuntimeError(f'Process exit {code}; see {log_path}')
                elif not (source_complete(item['output_dir']) if payload['stage'] == 'source'
                          else external_complete(item['output_dir'])):
                    raise RuntimeError(f'Process exited without complete output artifacts; see {log_path}')
                else:
                    save(identifier, status='complete', exit_code=code)
                    print('DONE', identifier, flush=True)
            except Exception as exc:
                if process is not None and process.poll() is None:
                    if not signal_group(job, signal.SIGTERM):
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        signal_group(job, signal.SIGKILL)
                        process.kill()
                        process.wait()
                reap_remaining_group(job)
                save(identifier, status='failed', error=str(exc))
                print('FAILED', identifier, str(exc), flush=True)
            finally:
                jobs.task_done()

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in previous_handlers:
        signal.signal(sig, lambda *_: stop_event.set())
    save()
    try:
        threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=.2)
        failures = sum(job['status'] == 'failed' for job in state['jobs'].values())
        state['status'] = 'stopped' if stop_event.is_set() else ('failed' if failures else 'complete')
        save()
        return 130 if stop_event.is_set() else int(failures > 0)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def stop(manifest, force=False):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Safe process-group stopping must run on the Linux server that launched the experiments.')
    state = read_state(manifest)
    if not state:
        print('No launcher record for this manifest.')
        return 0
    if state.get('manifest') != str(Path(manifest).resolve()) or state.get('manifest_sha256') != sha256(manifest):
        raise ValueError('Manifest changed since launch; restore the exact manifest before using stop.')
    parent = state.get('launcher')
    if same_process(parent):
        os.kill(parent['pid'], signal.SIGTERM)
    # Refresh after the dispatch loop has received its stop signal.
    time.sleep(.5)
    state = read_state(manifest)
    sent = 0
    for job in state.get('jobs', {}).values():
        sent += int(signal_group(job, signal.SIGKILL if force else signal.SIGTERM))
    if force and same_process(parent):
        os.kill(parent['pid'], signal.SIGKILL)
    deadline = time.monotonic() + (3 if force else 15)
    while time.monotonic() < deadline:
        current = read_state(manifest)
        if not same_process(parent) and not any(group_is_owned(job) for job in current.get('jobs', {}).values()):
            print(f'Stopped this manifest; signalled {sent} verified experiment groups.')
            return 0
        time.sleep(.3)
    print('Some verified processes remain. Re-run stop with --force if necessary.')
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    launch = sub.add_parser('run', help='One process per GPU, with completed jobs skipped.')
    launch.add_argument('--manifest', required=True)
    launch.add_argument('--gpus', nargs='+', required=True)
    launch.add_argument('--restart-incomplete', action='store_true')
    launch.add_argument('--dry-run', action='store_true')
    inspect = sub.add_parser('status', help='Read-only progress including latest epoch.')
    inspect.add_argument('--manifest', required=True)
    terminate = sub.add_parser('stop', help='Stop only the verified processes recorded for this manifest.')
    terminate.add_argument('--manifest', required=True)
    terminate.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command == 'status':
            status(args.manifest)
            return 0
        if args.command == 'stop':
            return stop(args.manifest, args.force)
        return run(args.manifest, args.gpus, args.restart_incomplete, args.dry_run)
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
