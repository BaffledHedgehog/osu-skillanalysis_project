"""Resumable, consent-scoped collection of public player data for shared ML."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import httpx

from .analyzer import MOD_BITS, ParsedMap, normalize_mods, parse_map
from .osu_api import PUBLIC_API, USER_AGENT
from .storage import Store

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = ROOT / '.data/population'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def confirmed_candidates(csv_path, consent_path):
    consent = read_json(consent_path, {})
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as f:
        candidates = list(csv.DictReader(f, delimiter=';'))
    if len({r['user_id'] for r in candidates}) != len(candidates):
        raise ValueError('Duplicate player IDs in candidates')
    allowed = {str(r['user_id']) for r in consent.get('players', []) if r.get('status') == 'granted'
               and 'public_scores_training_evaluation' in r.get('scopes', [])}
    # Closed by default: a public candidate list alone authorizes no collection.
    return [r for r in candidates if r['user_id'] in allowed]


class PublicFetcher:
    def __init__(self, client, interval=.4):
        self.client = client
        self.interval = interval
        self.lock = asyncio.Lock()
        self.next_at = 0.

    async def get(self, url, **kwargs):
        for attempt in range(3):
            async with self.lock:
                now = asyncio.get_running_loop().time()
                await asyncio.sleep(max(0, self.next_at-now))
                self.next_at = asyncio.get_running_loop().time() + self.interval
            try:
                response = await self.client.get(url, **kwargs)
                if response.status_code == 429:
                    try: delay = float(response.headers.get('Retry-After', '10'))
                    except ValueError: delay = 10
                    async with self.lock:
                        self.next_at = max(self.next_at, asyncio.get_running_loop().time()+min(60, max(1, delay)))
                elif response.status_code not in (500, 502, 503, 504):
                    response.raise_for_status()
                    return response
                response.raise_for_status()
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code not in (429,500,502,503,504):
                    raise
                if attempt == 2:
                    raise
                await asyncio.sleep(1 + attempt*2)
        raise RuntimeError('Unreachable retry state')


def normalize_score(score):
    """Never infer pass from accuracy, grade or best membership."""
    if score.get('mode_int',0) != 0 or score.get('mode','osu') != 'osu':
        raise ValueError('unsupported_mode')
    mods_data = score.get('mods') or []
    if isinstance(mods_data, int):
        allowed = sum(MOD_BITS[n] for n in ('NF','EZ','HD','HR','DT','HT','NC','FL'))
        if mods_data & ~allowed: raise ValueError('unsupported_mods')
        mods_data = [n for n, b in MOD_BITS.items() if mods_data & b]
    if any(isinstance(m, dict) and m.get('settings') for m in mods_data):
        raise ValueError('custom_mod_settings')
    mods, _ = normalize_mods([m.get('acronym', '') if isinstance(m, dict) else str(m) for m in mods_data])
    if 'NF' in mods: raise ValueError('no_fail')
    if type(score.get('passed')) is not bool: raise ValueError('unknown_outcome')
    accuracy = float(score.get('accuracy'))
    if not 0 <= accuracy <= 1: raise ValueError('invalid_accuracy')
    date = score.get('ended_at') or score.get('created_at')
    if not date: raise ValueError('missing_date')
    # Verify the timestamp but retain the source value unchanged.
    datetime.fromisoformat(date.replace('Z', '+00:00'))
    return mods, accuracy, date


async def collect_scores(folder=DEFAULT_FOLDER, candidates=ROOT/'deliverables/dataset-candidates.csv', force=False):
    folder = Path(folder)
    players = confirmed_candidates(candidates, folder/'consents.json')
    if not players: raise ValueError('No confirmed players in consent manifest')
    store = Store(folder/'population.db')
    results = read_json(folder/'players.json', {})
    semaphore = asyncio.Semaphore(2)
    done = 0
    async with httpx.AsyncClient(timeout=25, headers={'User-Agent': USER_AGENT}) as client:
        fetcher = PublicFetcher(client)

        async def collect(player):
            nonlocal done
            async with semaphore:
                uid = player['user_id']
                previous = results.get(uid, {})
                if previous.get('status') == 'complete' and not force:
                    done += 1
                    return
                record = {k: player[k] for k in ('user_id','username','pp_bin','pp','country')}
                record['sources'] = {}
                try:
                    profile_file = folder/'raw'/uid/'profile.json'
                    profile = read_json(profile_file) if not force else None
                    if profile is None:
                        profile = (await fetcher.get(f'{PUBLIC_API}/{uid}', params={'format': 'raw'})).json()
                        write_json(profile_file, profile)
                    if str(profile.get('id')) != uid: raise ValueError('profile_identity_mismatch')
                    user = profile['username']
                    record.update(username=user, current_pp=(profile.get('statistics') or {}).get('pp'))
                    store.save_public_profile(user, profile)
                    for source in ('recent', 'best', 'firsts'):
                        path = folder/'raw'/uid/f'{source}.json'
                        payload = read_json(path) if not force else None
                        try:
                            if payload is None:
                                params = {'type': source, 'mode': 'osu', 'limit': 100}
                                if source == 'recent': params['include_fails'] = 1
                                payload = (await fetcher.get(f'{PUBLIC_API}/{uid}/scores', params=params)).json()
                                rows = payload.get('scores') if isinstance(payload, dict) else payload
                                if not isinstance(rows, list): raise ValueError('invalid_score_list')
                                write_json(path, payload)
                            rows = payload.get('scores') if isinstance(payload, dict) else payload
                            rejected = Counter()
                            with store.batch():
                                for score in rows:
                                    if not isinstance(score, dict):
                                        rejected['invalid_row'] += 1; continue
                                    if score.get('user_id') is not None and str(score['user_id']) != uid:
                                        rejected['score_identity_mismatch'] += 1; continue
                                    key = str(score.get('legacy_score_id') or score.get('id') or hashlib.sha256(json.dumps(score, sort_keys=True).encode()).hexdigest())
                                    mid = str((score.get('beatmap') or {}).get('id') or score.get('beatmap_id') or '')
                                    reason = 'awaiting_chart'
                                    try: normalize_score(score)
                                    except (ValueError, TypeError) as exc:
                                        reason = str(exc) if isinstance(exc, ValueError) else 'invalid_fields'
                                        rejected[reason] += 1
                                    store.save_score_payload(user, 'hinai', source, key, mid, score, reason)
                            record['sources'][source] = {'received': len(rows), 'fails': sum(r.get('passed') is False for r in rows), 'excluded': dict(rejected)}
                        except (httpx.HTTPError, ValueError, KeyError):
                            record['sources'][source] = {'status': 'unavailable'}
                    record['status'] = 'complete' if all('received' in r for r in record['sources'].values()) else 'partial'
                except (httpx.HTTPError, ValueError, KeyError):
                    record['status'] = 'unavailable'
                record['fetched_at'] = datetime.now(timezone.utc).isoformat()
                results[uid] = record
                write_json(folder/'players.json', results)
                done += 1
                report = {'stage': 'scores', 'completed': done, 'total': len(players),
                          'players_by_status': dict(Counter(p.get('status') for p in results.values())),
                          'raw_rows': sum(sum(s.get('received',0) for s in p.get('sources',{}).values()) for p in results.values())}
                write_json(folder/'status.json', report)
                print(json.dumps({**report, 'last_user': record['username']}, ensure_ascii=True), flush=True)
        await asyncio.gather(*(collect(p) for p in players))
    return results


def seed_maps(store, source):
    if not Path(source).is_file() or Path(source).resolve() == store.path.resolve(): return
    with sqlite3.connect(source) as src, store.batch():
        for mid, checksum, content, metadata in src.execute('SELECT id,checksum,content,metadata FROM maps'):
            if store.get_map(mid) is None:
                store.save_map(ParsedMap(mid, checksum, content, json.loads(metadata)))


def normalize_population(folder=DEFAULT_FOLDER):
    """Rebuild derived labels from consent-scoped raw rows, checking chart revisions."""
    folder=Path(folder);store=Store(folder/'population.db')
    players=confirmed_candidates(ROOT/'deliverables/dataset-candidates.csv',folder/'consents.json')
    snapshots=read_json(folder/'players.json',{})
    accepted=[];rejected=Counter()
    # Reuse one connection for this disk-only pass; avoid reopening SQLite per score.
    with store.batch():
        for player in players:
            user=snapshots.get(player['user_id'],{}).get('username',player['username'])
            for row in store.score_payloads(user):
                score=row['payload']
                if score.get('user_id') is not None and str(score['user_id'])!=player['user_id']:
                    rejected['score_identity_mismatch']+=1;continue
                try:mods,accuracy,date=normalize_score(score)
                except (ValueError,TypeError):
                    rejected['unsupported_or_incomplete_label']+=1;continue
                parsed=store.get_map(row['map_id'])
                if not parsed:
                    rejected['chart_not_loaded']+=1;continue
                expected=(score.get('beatmap') or {}).get('checksum')
                if expected and parsed.checksum.casefold()!=str(expected).casefold():
                    rejected['chart_checksum_mismatch']+=1;continue
                source='recent' if 'recent' in row['sources'] else 'best' if 'best' in row['sources'] else 'firsts'
                stats=score.get('statistics') or {}
                accepted.append((user,row['map_id'],mods,accuracy,score['passed'],
                                 int(stats.get('miss',stats.get('count_miss',0)) or 0),date,
                                 f'hinai_{source}',f'osu-{row["score_key"]}',score.get('pp')))
    # Only the derived score index in this separate population DB is rebuilt.
    # All source payloads and charts are retained; the transaction is atomic.
    with store.batch():
        with store._connection() as db:db.execute('DELETE FROM plays')
        for row in accepted:store.save_play(*row)
    audit={'normalized':len(accepted),'by_source':dict(Counter(r[7] for r in accepted)),
           'ordinary_passes':sum(r[4] and r[7]=='hinai_recent' for r in accepted),
           'ordinary_fails':sum(not r[4] and r[7]=='hinai_recent' for r in accepted),
           'rejected':dict(rejected),'consented_players':len(players)}
    write_json(folder/'quality.json',audit)
    print(json.dumps(audit),flush=True)
    return audit


async def collect_maps(folder=DEFAULT_FOLDER, context_limit=30, force=False, recent_only_players=False):
    if context_limit < 0: raise ValueError('context_limit must be nonnegative')
    folder = Path(folder)
    players = confirmed_candidates(ROOT/'deliverables/dataset-candidates.csv', folder/'consents.json')
    results = read_json(folder/'players.json', {})
    store = Store(folder/'population.db')
    candidates = {}
    for player in players:
        user = results.get(player['user_id'], {}).get('username', player['username'])
        rows = store.score_payloads(user)
        ordinary, curated = [], []
        for row in rows:
            try: normalize_score(row['payload'])
            except (ValueError, TypeError): continue
            if not row['map_id'].isdigit(): continue
            (ordinary if 'recent' in row['sources'] else curated).append(row)
        # All ordinary attempts; a deterministic spread of best/firsts for historical context.
        if recent_only_players and not ordinary:
            continue
        curated.sort(key=lambda r: hashlib.sha256(r['score_key'].encode()).digest())
        for row in [*ordinary, *(curated if context_limit == 0 else curated[:context_limit])]:
            candidates[row['map_id']] = True
    errors = read_json(folder/'chart-errors.json', {})
    with store._connection() as db:
        existing_ids={r[0] for r in db.execute('SELECT id FROM maps')}
    pending = [m for m in candidates if m not in existing_ids and (force or m not in errors)]
    completed = 0
    ready=[]
    def flush():
        with store.batch():
            for parsed in ready:store.save_map(parsed)
        ready.clear()
    semaphore = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=20, headers={'User-Agent': USER_AGENT}, follow_redirects=False) as client:
        fetcher = PublicFetcher(client, interval=.25)
        async def chart(mid):
            nonlocal completed
            async with semaphore:
                try:
                    parsed=None
                    for url in (f'https://mirror.hinamizawa.ai/api/osu/{mid}/raw',f'https://osu.ppy.sh/osu/{mid}'):
                        try:
                            response=await fetcher.get(url)
                            if len(response.content)>8_000_000:raise ValueError('chart_too_large')
                            candidate=parse_map(response.content)
                            if candidate.map_id.isdigit() and candidate.map_id!=mid:raise ValueError('wrong_chart_id')
                            parsed=ParsedMap(mid,candidate.checksum,candidate.content,candidate.metadata)
                            break
                        except (httpx.HTTPError,ValueError):
                            continue
                    if parsed is None:raise ValueError('chart_unavailable')
                    ready.append(parsed)
                    if len(ready)>=20:flush()
                except (httpx.HTTPError, ValueError):
                    errors[mid] = 'download_or_parse_failed'
                completed += 1
                if completed % 25 == 0 or completed == len(pending):
                    write_json(folder/'chart-errors.json', errors)
                    status = {'stage':'charts','completed':completed,'pending_at_start':len(pending),
                              'selected_maps':len(candidates),'failures':len(errors),'context_limit':context_limit,
                              'recent_only_players':recent_only_players}
                    write_json(folder/'status.json', status)
                    print(json.dumps(status), flush=True)
        await asyncio.gather(*(chart(mid) for mid in pending))
    flush()
    normalized = Counter()
    for player in players:
        user = results.get(player['user_id'], {}).get('username', player['username'])
        with store.batch():
            for row in store.score_payloads(user):
                score = row['payload']
                try: mods, accuracy, date = normalize_score(score)
                except (ValueError, TypeError): continue
                mid = row['map_id']
                if not store.get_map(mid): continue
                source = 'recent' if 'recent' in row['sources'] else 'best' if 'best' in row['sources'] else 'firsts'
                stats = score.get('statistics') or {}
                changed = store.save_play(user, mid, mods, accuracy, score['passed'],
                                          int(stats.get('miss',stats.get('count_miss',0)) or 0), date,
                                          f'hinai_{source}', f'osu-{row["score_key"]}', score.get('pp'), grade=score.get('rank'))
                normalized[source] += 1
    result = {'stage':'ready','players':len(players),'selected_maps':len(candidates),
              'normalized_by_source':dict(normalized),'chart_failures':len(errors),'context_limit':context_limit,
              'recent_only_players':recent_only_players,
              'note':'Raw scores retained in full. Chart subset is all usable recent plus deterministic curated context.'}
    write_json(folder/'status.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('scores','charts','normalize','all'), default='all')
    p.add_argument('--folder', type=Path, default=DEFAULT_FOLDER)
    p.add_argument('--context-limit', type=int, default=30, help='0 downloads all usable best/firsts maps')
    p.add_argument('--force', action='store_true')
    p.add_argument('--recent-only-players', action='store_true', help='Pilot: fetch context charts only for players with usable recent targets')
    args=p.parse_args()
    if args.phase in ('scores','all'): asyncio.run(collect_scores(args.folder, force=args.force))
    if args.phase in ('charts','all'): asyncio.run(collect_maps(args.folder, args.context_limit, args.force, args.recent_only_players))
    if args.phase in ('normalize','all'): normalize_population(args.folder)


if __name__ == '__main__': main()
