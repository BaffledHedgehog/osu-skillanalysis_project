"""Local training jobs and gated CNN inference, separate from request-time fitting."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

import numpy as np
import psutil

from .ml import BEST_SOURCES, FEATURES as GLOBAL_FEATURES, vector
from .recency import HALF_LIFE_DAYS, played_date
from .sequence import FEATURES, VERSION, windows
from .sequence_data import CONTEXT_FEATURES, SequenceCache, validated_mods
from .storage import Store

ROOT = Path(__file__).resolve().parents[1]
_JOB_LOCK = threading.Lock()


def model_folder(store, user):
    key = hashlib.sha256(f'{store.path.resolve()}:{user.strip().casefold()}'.encode()).hexdigest()[:24]
    return store.path.parent / 'cnn' / key


def _json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def _runs(store, user, local=False):
    root = model_folder(store, user) / ('local-runs' if local else 'runs')
    def activity(path):
        return max((p.stat().st_mtime for p in (path/'report.json',path/'progress.json') if p.is_file()),default=0)
    return sorted((p for p in root.glob('*') if p.is_dir()), key=activity, reverse=True)


def _running_job():
    # Inspect live processes, never infer liveness from a progress file alone.
    for process in psutil.process_iter(['pid', 'cmdline', 'cwd']):
        try:
            command = process.info['cmdline'] or []
            if '--audit' not in command and any(m in command for m in ('app.sequence_train', 'app.sequence_local', 'app.replay_events')) and Path(process.info['cwd'] or '').resolve() == ROOT:
                return process
        except (psutil.Error, OSError):
            continue
    return None


def status(store: Store, user: str, local=False):
    runs = _runs(store, user, local)
    report_run = next((p for p in runs if _json(p/'report.json')), None)
    report = _json(report_run/'report.json') if report_run else None
    latest = _json(runs[0] / 'progress.json', {}) if runs else {}
    process = _running_job()
    active = False
    if process:
        try:
            command = process.cmdline()
            output = command[command.index('--output') + 1]
            running_path = (ROOT / output).resolve()
            active = any(p.resolve() == running_path for p in runs)
        except (psutil.Error, ValueError, IndexError, OSError):
            pass
    if latest.get('status') == 'running' and not active:
        latest = {**latest, 'status': 'interrupted', 'message': 'Процесс обучения завершился без отчета. Можно запустить заново.'}
    if active and local and latest.get('phase') in ('starting', 'replays'):
        path = store.path.parent / 'replay-events' / ('collection-' + hashlib.sha256(user.strip().casefold().encode()).hexdigest()[:16] + '.json')
        collection = _json(path, {})
        latest = {**latest, 'phase': 'replays', 'step': collection.get('processed', 0),
                  'total': collection.get('requested', 120), 'verified_replays': collection.get('verified', 0)}
    return {'available': importlib.util.find_spec('torch') is not None, 'running': active,
            'progress': latest, 'report': report, 'forward': _json(report_run/'forward.json') if report_run else None,
            'status': 'running' if active else 'trained' if report and any(r['accepted'] for r in report['release'].values()) else 'rejected' if report else 'not_trained'}


def check_forward(store,user):
    if not importlib.util.find_spec('torch'):
        raise ValueError('Установите PyTorch: python -m pip install -r requirements-cnn.txt')
    run=next((p for p in _runs(store,user) if (p/'report.json').is_file()),None)
    if run is None:raise ValueError('Сначала обучите и сохраните CNN.')
    from .sequence_forward import evaluate_forward
    return evaluate_forward(store,user,run)


def start_training(store: Store, user: str, local=False):
    if not importlib.util.find_spec('torch'):
        raise ValueError('Установите PyTorch: python -m pip install -r requirements-cnn.txt')
    if store.play_count(user) < (30 if local else 80):
        raise ValueError('Для CNN нужны хотя бы 80 обычных попыток с известным исходом.')
    with _JOB_LOCK:
        if _running_job():
            raise ValueError('Обучение CNN уже идет. Дождитесь его завершения.')
        run = model_folder(store, user) / ('local-runs' if local else 'runs') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        run.mkdir(parents=True)
        command = [sys.executable, '-m', 'app.sequence_local' if local else 'app.sequence_train', user, '--database', str(store.path.resolve()),
                   '--output', str(run.resolve()), '--steps', '400', '--pretrain-steps', '200']
        command.extend(['--collect', '120'] if local else ['--batch-size', '8'])
        (run / 'progress.json').write_text(json.dumps({'status': 'running', 'phase': 'starting', 'step': 0, 'total': 1}), encoding='utf-8')
        with (run / 'worker.log').open('wb') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                       env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        return {'started': True, 'run_id': run.name}


def stop_training(store, user):
    """Only stop this user's worker for this database, plus its own children."""
    process = _running_job()
    if not process: return {'stopped': False}
    try:
        command = process.cmdline()
        if '--output' not in command: return {'stopped': False}
        output = (ROOT / command[command.index('--output') + 1]).resolve()
        root = model_folder(store, user).resolve()
        if output.parent not in (root / 'runs', root / 'local-runs'):
            return {'stopped': False}
        children = process.children(recursive=True)
        process.terminate()
        for child in children:
            try: child.terminate()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs([process, *children], timeout=3)
        for remaining in alive:
            try: remaining.kill()
            except psutil.NoSuchProcess: pass
        _, alive = psutil.wait_procs(alive, timeout=2)
        if alive: return {'stopped': False}
        return {'stopped': True}
    except (psutil.Error, ValueError, OSError, IndexError):
        return {'stopped': False}


def current_profile(records, now=None):
    now = now or datetime.now(timezone.utc)
    count = passed = ac_count = ac_sum = stars = aim = speed = ar = hd_count = hd_passes = 0.
    for p in records:
        date = played_date(p)
        if not date or date >= now or p['source'] in BEST_SOURCES or not p.get('passed_known', 1) or 'NF' in p['mods']:
            continue
        weight = .5 ** ((now - date).total_seconds() / 86400 / HALF_LIFE_DAYS)
        success = float(p['passed'])
        a = p['analysis']
        count += weight; passed += weight * success
        if success and p.get('complete', 1):
            ac_count += weight; ac_sum += weight * p['accuracy']
        stars += weight * success * a['stars']; aim += weight * success * a['aim']
        speed += weight * success * a['speed']; ar += weight * success * a['ar']
        if 'HD' in p['mods']:
            hd_count += weight; hd_passes += weight * success
    import math
    return [math.log1p(count), (passed + 1) / (count + 2), ac_sum / ac_count if ac_count else .9,
            stars / passed if passed else 0, aim / passed if passed else 0, speed / passed if passed else 0,
            ar / passed if passed else 0, (hd_passes + 1) / (hd_count + 2)]


@lru_cache(maxsize=8)
def _load(run_path, variant, stamp):
    import torch
    from .sequence_net import PatternModel, load_weights
    run = Path(run_path)
    artifact = _json(run / 'artifact.json')
    if (artifact.get('sequence_version', 'note-sequence-v1') != VERSION or artifact['features'] != list(FEATURES)
            or artifact['context_features'] != list(CONTEXT_FEATURES)):
        raise ValueError('CNN feature schema changed; retraining required.')
    weight_path = run / f'{variant}.npz'
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != artifact['weight_hashes'][variant]:
        raise ValueError('CNN weights failed checksum verification.')
    model = PatternModel(**artifact['model_options'], sequence=True)
    load_weights(model, weight_path)
    model.eval()
    # CPU inference leaves the GPU available for training and the osu! client.
    torch.set_num_threads(min(2, torch.get_num_threads()))
    return artifact, model


def _linear(model, features):
    return float(model['intercept'] + np.sum((np.asarray(features) - model['mean']) / model['scale'] * model['coef']))


@lru_cache(maxsize=64)
def _active_run(root, bucket):
    paths = sorted(Path(root).glob('runs/*/report.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        if not (path.parent / 'snapshot/snapshot.json').is_file():
            continue
        report = _json(path)
        if report and any(v['accepted'] for v in report['release'].values()):
            return str(path.parent)
    return None


@lru_cache(maxsize=16)
def _metadata(run_path, stamp):
    run = Path(run_path)
    report, artifact = _json(run / 'report.json'), _json(run / 'artifact.json')
    if not artifact or not report:
        return None
    manifest, samples = _json(run / 'manifest.json'), _json(run / 'snapshot/samples.json')
    supported_mods = set(artifact['support']['mods'])
    head_mods = artifact['support'].get('validated_mods', {'pass': [], 'accuracy': []})
    bounds = artifact['support']['stars']
    if samples and manifest:
        train = manifest['splits']['train']['indices']
        supported_mods = {samples[i]['mods'] for i in train}
        map_index = _json(run / 'snapshot/maps.json', {})
        families = {mid: item['family'] for mid, item in map_index.items()}
        head_mods = validated_mods(samples, families, manifest['splits']['test']['indices'])
        train_families = set(manifest['splits']['train']['families'])
        reference = tuple(sorted({item['source_checksum'] for item in map_index.values() if item['family'] in train_families}))
        with np.load(run / 'snapshot/inputs.npz', allow_pickle=False) as inputs:
            stars = inputs['context'][train, 0]
            bounds = [float(stars.min()), float(stars.max())]
    else:
        reference = ()
    return report, artifact, supported_mods, bounds, head_mods, reference


@lru_cache(maxsize=64)
def _profile(database, user, revision, bucket, reference=()):
    records = Store(database).plays_with_analysis(user)
    if reference:
        checksums = set(reference)
        records = [p for p in records if p.get('analysis') and p['analysis']['checksum'] in checksums]
    return current_profile(records)


def predict_cnn(store: Store, user: str, target: dict, records):
    try:
        run_name = _active_run(str(model_folder(store, user)), int(time.time() // 10))
        if run_name is None:
            return {'status': 'not_trained', 'pass_probability': None, 'accuracy': None}
        stamp = (Path(run_name) / 'artifact.json').stat().st_mtime_ns
        return _predict_cached(str(store.path.resolve()), user.strip().casefold(), store.cache_token(user),
                               target['id'], target['mods'], target.get('checksum'), int(time.time() // 600),
                               run_name, stamp)
    except (OSError, ValueError, KeyError, RuntimeError):
        # An unusable optional checkpoint must not break the ordinary predictor.
        return {'status': 'invalid_artifact', 'pass_probability': None, 'accuracy': None}


@lru_cache(maxsize=1024)
def _predict_cached(database, user, revision, map_id, mods, checksum, profile_bucket, run_name, stamp):
    store = Store(database)
    target = store.analysis(map_id, mods)
    if target is None:
        return {'status': 'missing_map', 'pass_probability': None, 'accuracy': None}
    run = Path(run_name)
    meta = _metadata(run_name, stamp)
    if meta is None:
        return {'status': 'invalid_artifact', 'pass_probability': None, 'accuracy': None}
    report, artifact, supported_mods, bounds, head_mods, reference = meta
    if target['mods'] not in supported_mods or not any(target['mods'] in head_mods[h] and artifact['release'][h]['accepted'] for h in ('pass', 'accuracy')):
        return {'status': 'insufficient_mod_data', 'pass_probability': None, 'accuracy': None}
    lower, upper = bounds
    if not lower <= target['stars'] <= upper:
        return {'status': 'out_of_range', 'pass_probability': None, 'accuracy': None}
    if not importlib.util.find_spec('torch'):
        return {'status': 'missing_dependency', 'pass_probability': None, 'accuracy': None}
    import torch
    parsed = store.get_map(target['id'])
    if not parsed:
        return {'status': 'missing_map', 'pass_probability': None, 'accuracy': None}
    sequence = SequenceCache(store).get(parsed, target)
    chunks, masks, starts = windows(sequence)
    raw = vector(target)
    profile = _profile(str(store.path.resolve()), user.strip().casefold(), store.cache_token(user), int(time.time() // 600), reference)
    context = np.asarray([*raw, *profile], dtype=np.float32)
    context = ((context - artifact['context_mean']) / artifact['context_scale']).astype(np.float32)
    base_p = _linear(artifact['baseline']['pass'], raw)
    base_a = float(np.clip(_linear(artifact['baseline']['accuracy'], raw), 0, 1))
    result = {'status': 'trained', 'pass_probability': None, 'accuracy': None, 'run_id': run.name,
              'release': artifact['release']}
    completed_variants = {}
    for head in ('pass', 'accuracy'):
        release = artifact['release'][head]
        if not release['accepted'] or target['mods'] not in head_mods[head]:
            continue
        variant = release['variant']
        if variant not in completed_variants:
            _, model = _load(str(run.resolve()), variant, (run / 'artifact.json').stat().st_mtime_ns)
            with torch.inference_mode():
                encoded = []
                for i in range(0, len(chunks), 64):
                    values = torch.from_numpy(chunks[i:i + 64])
                    mask = torch.from_numpy(masks[i:i + 64])
                    part, _ = model.encode_chunks(values, mask)
                    encoded.append(part)
                encoded = torch.cat(encoded)
                logits, accuracy = model.from_chunks(encoded, torch.zeros(len(encoded), dtype=torch.long),
                    torch.from_numpy(context[None]), torch.tensor([base_p]), torch.tensor([base_a]))
                completed_variants[variant] = (float(torch.sigmoid(logits)[0]), float(accuracy[0]))
        p, a = completed_variants[variant]
        if head == 'pass': result['pass_probability'] = p
        else:
            result['accuracy'] = a
            result['accuracy_error_q90'] = report['variants'][variant]['test']['accuracy_error_q90']
            result['accuracy_metrics'] = report['variants'][variant]['test']
    return result


@lru_cache(maxsize=8)
def _load_local(run_path, variant, stamp):
    from .sequence_net import LocalPatternModel, load_weights
    from .sequence_labels import TARGETS
    run = Path(run_path)
    artifact = _json(run / 'artifact.json')
    if (artifact['schema'] != 'local-pattern-cnn-v1' or artifact.get('sequence_version', 'note-sequence-v1') != VERSION or artifact['features'] != list(FEATURES)
            or artifact['context_features'] != list(CONTEXT_FEATURES) or artifact['targets'] != list(TARGETS)):
        raise ValueError('Local CNN schema changed; retraining required.')
    path = run / f'{variant}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['weight_hashes'][variant]:
        raise ValueError('Local CNN weights failed checksum verification.')
    return artifact, load_weights(LocalPatternModel(**artifact['model_options']), path).eval()


def predict_local(store, user, target):
    if not user: return {'status': 'not_trained', 'windows': []}
    try:
        runs = _runs(store, user, local=True)
        if not runs: return {'status': 'not_trained', 'windows': []}
        run = next((p for p in runs if (p/'report.json').is_file() and (p/'artifact.json').is_file()), None)
        if run is None: return {'status': 'not_trained', 'windows': []}
        report, artifact = _json(run/'report.json'), _json(run/'artifact.json')
        if not any(v['accepted'] for v in artifact['release'].values()):
            return {'status': 'rejected', 'release': report['release'], 'windows': []}
        eligible = artifact['support'].get('validated_mods', {})
        if target['mods'] not in artifact['support']['mods'] or not any(target['mods'] in eligible.get(t, []) and artifact['release'][t]['accepted'] for t in artifact['release']):
            return {'status': 'insufficient_mod_data', 'windows': []}
        lo, hi = artifact['support']['stars']
        if not lo <= target['stars'] <= hi:
            return {'status': 'out_of_range', 'windows': []}
        if not importlib.util.find_spec('torch'):
            return {'status': 'missing_dependency', 'windows': []}
        return _predict_local_cached(str(store.path.resolve()), user.casefold(), store.cache_token(user),
            target['id'], target['mods'], target.get('checksum'), int(time.time()//600), str(run.resolve()),
            (run/'artifact.json').stat().st_mtime_ns)
    except (OSError, ValueError, KeyError, RuntimeError):
        return {'status': 'invalid_artifact', 'windows': []}


@lru_cache(maxsize=128)
def _predict_local_cached(database, user, revision, map_id, mods, checksum, bucket, run_name, stamp):
    import torch
    from .sequence_local import predict_notes
    from .sequence_labels import TARGETS
    torch.set_num_threads(min(2, torch.get_num_threads()))
    store = Store(database)
    target, parsed = store.analysis(map_id, mods), store.get_map(map_id)
    if not target or not parsed: return {'status': 'missing_map', 'windows': []}
    sequence = SequenceCache(store).get(parsed, target)
    outputs, variants = {}, {}
    artifact = _json(Path(run_name)/'artifact.json')
    manifest = _json(Path(run_name)/'manifest.json')
    if 'profile_reference' not in manifest:
        raise ValueError('Local checkpoint lacks the training profile reference.')
    raw = np.asarray([*vector(target), *_profile(database,user,revision,bucket,tuple(manifest['profile_reference']))], dtype=np.float32)
    for k, name in enumerate(TARGETS):
        release = artifact['release'][name]
        if not release['accepted'] or mods not in artifact['support']['validated_mods'].get(name, []): continue
        variant = release['variant']
        if variant not in variants:
            _, model = _load_local(run_name,variant,stamp)
            context = ((raw-artifact['context_mean'])/artifact['context_scale']).astype(np.float32)
            variants[variant] = predict_notes(model,sequence,context,raw,artifact['baseline'])
        outputs[name] = variants[variant][:, k]
    _, masks, starts = windows(sequence)
    points = []
    for j, pos in enumerate(starts):
        n = int(masks[j].sum()); end = int(pos)+n
        points.append({'start':round(float(sequence.times_ms[pos])/1000,2),
            'end':round(float(sequence.end_times_ms[end-1])/1000,2),'notes':n,
            **{name:round(float(value[pos:end].mean())*100,2) for name,value in outputs.items()}})
    return {'status':'trained','windows':points,'release':artifact['release'],
        'message':'Ожидаемая доля нот с ошибкой в участке по полным сохраненным реплеям. Это не вероятность провала участка и не оценка причины ошибки.'}
