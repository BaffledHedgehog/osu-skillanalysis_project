"""Verified judgement events from full local stable/lazer replays.

python -m app.replay_events USER --limit 120 --workers 2
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time
import uuid

from .replay_health import ROOT, _engine, _RUN_LOCK
from .storage import Store

SCHEMA = 'osu-2026.916.0-judgements-v1'


def _atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def event_key(play, checksum):
    return hashlib.sha256(f'{SCHEMA}:{play["sha256"]}:{checksum}'.encode()).hexdigest()


def extract_events(store: Store, user: str, play_id: str):
    play = store.replay_input(user, play_id)
    if not play or not play.get('complete', 1):
        raise ValueError('Для полной разметки нужен реплей законченной карты. Неигранные ноты не считаются промахами.')
    path = Path(play['path'])
    if not path.is_file() or path.stat().st_size > 25_000_000:
        raise ValueError('Реплей отсутствует или превышает 25 МБ.')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != play['sha256']:
        raise ValueError('Реплей изменился. Повторите локальный импорт.')
    parsed = store.get_map(play['map_id'])
    if parsed is None:
        raise ValueError('Нет точной карты реплея.')
    key = event_key(play, parsed.checksum)
    cache = store.path.parent / 'replay-events' / (key + '.json')
    if cache.exists():
        data = json.loads(cache.read_text(encoding='utf-8'))
        if data.get('schema') == SCHEMA and data.get('replay_sha256') == play['sha256']:
            return {**data, 'cached': True}
    chart = parsed.content.encode('utf-8')
    if hashlib.md5(chart).hexdigest() != parsed.checksum:
        chart = b'\xef\xbb\xbf' + chart
    if hashlib.md5(chart).hexdigest() != parsed.checksum:
        raise ValueError('Не удалось восстановить точные байты карты.')
    # Serialise compilation with the existing HP worker. Each run has its own cwd.
    with _RUN_LOCK:
        dotnet, binary = _engine()
    base = ROOT / '.data/event-runs'
    base.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix='events-', dir=base) as temp:
        folder = Path(temp)
        source, replay, output = folder / 'map.osu', folder / 'replay.osr', folder / 'result.json'
        source.write_bytes(chart); replay.write_bytes(raw)
        try:
            process = subprocess.run([dotnet, str(binary), str(source), str(replay), str(output), '--events'],
                cwd=folder, capture_output=True, timeout=180,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                env={**os.environ, 'DOTNET_CLI_TELEMETRY_OPTOUT': '1'})
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Разбор реплея превысил 3 минуты.') from exc
        if process.returncode or not output.is_file():
            raise ValueError('Движок не смог разобрать реплей.')
        result = json.loads(output.read_text(encoding='utf-8-sig'))
    if result.get('status') not in ('verified', 'statistics_mismatch'):
        raise ValueError('Разбор реплея завершился ошибкой.')
    # Native verification requires matching totals, no unmapped/reverted results,
    # and at least one judgement for every root object. Incomplete or mismatched
    # replays cannot silently become a full sequence of failure labels.
    data = {'schema': SCHEMA, 'play_id': play_id, 'map_id': play['map_id'], 'mods': play['mods'],
            'map_checksum': parsed.checksum, 'replay_sha256': play['sha256'],
            'verified': bool(result.get('events_verified')), 'complete': True,
            'elapsed_seconds': round(time.time() - started, 2), 'engine': result, 'cached': False}
    _atomic_json(cache, data)
    return data


def collect_events(store, user, *, limit=120, workers=2):
    plays = store.plays_with_analysis(user)
    # A deterministic sample of distinct map/mod pairs. Never pick by outcome,
    # accuracy or miss count; selection bias would distort local error rates.
    rows, seen = [], set()
    for p in plays:
        key = (p['map_id'], p['mods'])
        if key in seen or not p.get('complete', 1) or not store.replay_input(user, p['id']):
            continue
        seen.add(key); rows.append(p)
    random.Random(2026).shuffle(rows)
    rows = rows[:limit]
    folder = store.path.parent / 'replay-events'
    job = folder / ('collection-' + hashlib.sha256(user.strip().casefold().encode()).hexdigest()[:16] + '.json')
    report = {'schema': SCHEMA, 'status': 'running', 'requested': len(rows), 'processed': 0, 'verified': 0,
              'policy': 'latest full replay per map/mod, seeded sample; no outcome-based selection', 'results': []}
    _atomic_json(job, report)
    def run(p):
        try:
            result = extract_events(store, user, p['id'])
            return {'play_id': p['id'], 'map_id': p['map_id'], 'mods': p['mods'],
                    'verified': result['verified'], 'key': event_key(store.replay_input(user, p['id']), result['map_checksum']),
                    'cached': result['cached'], 'events': len(result['engine'].get('events') or [])}
        except (ValueError, OSError) as exc:
            return {'play_id': p['id'], 'verified': False, 'error': str(exc)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run, rows):
            report['results'].append(result)
            report['processed'] += 1; report['verified'] += int(result['verified'])
            _atomic_json(job, report)
            print(json.dumps({k: report[k] for k in ('processed', 'requested', 'verified')}), flush=True)
    report['status'] = 'completed'
    _atomic_json(job, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('user'); parser.add_argument('--database', default='.data/osu_skill.db')
    parser.add_argument('--limit', type=int, default=120); parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.limit <= 2000 or not 1 <= args.workers <= 2:
        parser.error('Allowed: limit 1..2000, workers 1..2')
    collect_events(Store(args.database), args.user, limit=args.limit, workers=args.workers)
