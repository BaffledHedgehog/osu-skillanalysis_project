"""Read osu!lazer's Realm snapshot with the official Realm .NET library."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from collections import Counter

from .analyzer import normalize_mods, parse_map
from .storage import Store

_BUILD_LOCK = threading.Lock()
READER_PROJECT = Path(__file__).resolve().parent.parent / 'tools/LazerReader/LazerReader.csproj'


def _reader():
    dotnet = shutil.which('dotnet')
    if not dotnet:
        raise ValueError('Для первого импорта osu!lazer нужен .NET SDK 8 или новее. После установки повторите импорт.')
    output = READER_PROJECT.parents[2] / '.data/tools/lazer-reader'
    binary = output / 'LazerReader.dll'
    with _BUILD_LOCK:
        source_mtime = max(p.stat().st_mtime for p in READER_PROJECT.parent.glob('*') if p.suffix in ('.cs', '.csproj'))
        if not binary.exists() or binary.stat().st_mtime < source_mtime:
            result = subprocess.run([dotnet, 'build', str(READER_PROJECT), '-c', 'Release', '-o', str(output), '--nologo'],
                                    capture_output=True, timeout=180,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            if result.returncode:
                raise ValueError('Не удалось собрать модуль Realm. Проверьте .NET SDK и доступ к NuGet, затем повторите импорт.')
    return dotnet, binary


def _read_realm(root: Path):
    database = root / 'client.realm'
    before = database.stat()
    if before.st_size > 512_000_000:
        raise ValueError('client.realm превышает лимит 512 МБ для локального импорта.')
    dotnet, binary = _reader()
    with tempfile.TemporaryDirectory(prefix='pulse-realm-') as temp:
        snapshot = Path(temp) / 'snapshot.realm'
        shutil.copyfile(database, snapshot)
        after = database.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('osu!lazer изменил базу во время копирования. Закройте клиент и повторите импорт.')
        result = subprocess.run([dotnet, str(binary), str(snapshot)], capture_output=True, timeout=90,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            raise ValueError('Не удалось прочитать копию client.realm. Закройте lazer и повторите; при новой версии клиента может потребоваться обновление модуля.')
        try:
            return json.loads(result.stdout)['scores']
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError('Модуль Realm вернул неизвестный формат результатов.') from exc


def stored_file(root: Path, digest: str) -> Path | None:
    if not isinstance(digest, str) or not re.fullmatch('[a-fA-F0-9]{64}', digest):
        return None
    digest = digest.lower()
    path = root / 'files' / digest[0] / digest[:2] / digest
    return path if path.is_file() else None


def import_lazer_scores(store: Store, root: Path, user=None, force=False):
    # One transaction avoids thousands of disk commits on large local histories.
    with store.batch():
        return _import_lazer_scores(store, root, user, force)


def _import_lazer_scores(store: Store, root: Path, user=None, force=False):
    info = (root / 'client.realm').stat()
    key = f'lazer-rank-v3:{root}:{(user or "").casefold()}:{info.st_size}:{info.st_mtime_ns}'
    if not force:
        cached = store.cached_import(key, ttl=None)
        if cached:
            return {**cached, 'cached': True, 'imported': 0}
    rows = _read_realm(root)
    players = Counter(r['user'] for r in rows if r.get('mode') == 0 and r.get('user'))
    user = user or (players.most_common(1)[0][0] if players else '')
    if not user:
        raise ValueError('В client.realm пока нет сохраненных результатов osu!standard.')
    selected = [r for r in rows if r.get('mode') == 0 and (r.get('user') or '').casefold() == user.casefold()]
    if not selected:
        raise ValueError('В osu!lazer нет результатов выбранного игрока.')
    imported = skipped = incomplete = replays = known = no_fail = 0
    missing, seen_maps, unsupported = set(), {}, Counter()
    for row in selected:
        try:
            mods_data = json.loads(row.get('mods') or '[]')
            if not isinstance(mods_data, list) or any(mod.get('settings') for mod in mods_data):
                skipped += 1
                continue
            acronyms = [mod.get('acronym', '').upper() for mod in mods_data]
            try:
                mods, _ = normalize_mods(acronyms)
            except ValueError:
                for acronym in acronyms:unsupported[acronym] += 1
                skipped += 1
                continue
            digest = row.get('map_file_hash') or row.get('map_hash')
            if digest not in seen_maps:
                filename = stored_file(root, digest)
                if not filename or filename.stat().st_size > 8_000_000:
                    missing.add(digest)
                    continue
                raw = filename.read_bytes()
                if hashlib.sha256(raw).hexdigest() != digest.lower():
                    missing.add(digest)
                    continue
                parsed = parse_map(raw)
                previous = store.get_map_by_checksum(parsed.checksum)
                if previous:
                    parsed = previous
                elif store.get_map(parsed.map_id):
                    parsed = parse_map(raw, f'lazer-{digest}')
                store.save_map(parsed)
                seen_maps[digest] = parsed
            parsed = seen_maps[digest]
            stats = {k.casefold(): v for k, v in json.loads(row.get('statistics') or '{}').items()}
            total = sum(int(stats.get(k, 0)) for k in ('great', 'ok', 'meh', 'miss'))
            complete = total == parsed.metadata['object_count']
            rank = row.get('rank')
            grade = {-1: 'F', 0: 'D', 1: 'C', 2: 'B', 3: 'A', 4: 'S', 5: 'SH', 6: 'X', 7: 'XH'}.get(rank)
            nf = 'NF' in acronyms
            passed_known = grade is not None and not nf
            if not complete:
                incomplete += 1
            # A partial result is useful when Rank explicitly confirms failure.
            # An ungraded truncated replay may be an abort; do not label it a fail.
            if not total or (not complete and grade != 'F'):
                skipped += 1
                continue
            accuracy = float(row['accuracy'])
            if not 0 <= accuracy <= 1:
                skipped += 1
                continue
            legacy_id, online_id = row.get('legacy_online_id', -1), row.get('online_id', -1)
            identity = f'osu-{legacy_id}' if legacy_id > 0 else f'osu-{online_id}' if online_id > 0 else f'lazer-{row["id"]}'
            imported += int(store.save_play(user, parsed.map_id, mods, accuracy, grade not in (None, 'F'), int(stats.get('miss', 0)),
                                           row.get('played_at'), 'local_lazer', identity, passed_known=passed_known,
                                           complete=complete, grade=grade))
            known += int(passed_known)
            no_fail += int(nf)
            for file in row.get('replay_files', []):
                replay_path = stored_file(root, file.get('hash'))
                if replay_path and str(file.get('name', '')).lower().endswith('.osr'):
                    store.save_local_replay(identity, replay_path, file['hash'])
                    break
            replays += sum(bool(stored_file(root, file.get('hash'))) for file in row.get('replay_files', []))
        except (ValueError, TypeError, KeyError, OSError):
            skipped += 1
    result = {'username': user, 'provider': 'local_lazer', 'imported': imported,
              'total_plays': store.play_count(user), 'local_scores': len(selected), 'skipped': skipped,
              'missing_maps': len(missing), 'incomplete_scores': incomplete, 'available_replays': replays,
              'unsupported_mods': dict(unsupported), 'local_players': list(players), 'cached': False,
              'known_outcomes': known, 'no_fail_scores': no_fail,
              'highlight_map_id': next((m.map_id for m in seen_maps.values()), None),
              'message': 'Результат прохождения прочитан из Rank: F - провал. Неполные попытки с F сохранены. No Fail хранится отдельно и не считается обычным pass.'}
    store.cache_import(key, result)
    return result
