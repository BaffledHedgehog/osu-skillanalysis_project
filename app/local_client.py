"""Read osu!stable results from an explicitly requested local installation."""
from __future__ import annotations

import hashlib
import os
import re
import struct
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .analyzer import MOD_BITS, normalize_mods, parse_map
from .storage import Store

MAX_DB_BYTES = 256_000_000
MAX_SCORES = 500_000


class Reader:
    def __init__(self, data: bytes):
        self.data, self.offset = memoryview(data), 0

    def number(self, fmt):
        size = struct.calcsize('<' + fmt)
        if self.offset + size > len(self.data):
            raise ValueError('scores.db обрезан или записывается клиентом. Повторите импорт позже.')
        value = struct.unpack_from('<' + fmt, self.data, self.offset)[0]
        self.offset += size
        return value

    def string(self):
        marker = self.number('B')
        if marker == 0:
            return ''
        if marker != 11:
            raise ValueError('Неизвестный формат строки в scores.db.')
        length = 0
        for i in range(5):
            byte = self.number('B')
            length |= (byte & 127) << (7 * i)
            if not byte & 128:
                break
        else:
            raise ValueError('Поврежденная длина строки в scores.db.')
        if length > 1_000_000 or self.offset + length > len(self.data):
            raise ValueError('Поврежденная строка в scores.db.')
        value = bytes(self.data[self.offset:self.offset + length]).decode('utf-8', errors='strict')
        self.offset += length
        return value


def read_scores(data: bytes):
    if not data or len(data) > MAX_DB_BYTES:
        raise ValueError('scores.db пустой или превышает 256 МБ.')
    r = Reader(data)
    version, map_count = r.number('i'), r.number('i')
    if not 20140721 <= version <= 21000101 or not 0 <= map_count <= MAX_SCORES:
        raise ValueError('Эта версия scores.db не поддерживается или файл поврежден.')
    rows = []
    for _ in range(map_count):
        group_hash, count = r.string(), r.number('i')
        if not 0 <= count <= MAX_SCORES - len(rows):
            raise ValueError('Некорректное число результатов в scores.db.')
        for _ in range(count):
            mode, client_version = r.number('B'), r.number('i')
            checksum, user, replay_hash = r.string(), r.string(), r.string()
            counts = [r.number('H') for _ in range(6)]
            score, combo, perfect, mods = r.number('i'), r.number('H'), r.number('B'), r.number('i')
            r.string()  # Empty legacy life-bar slot; no account credentials are read.
            ticks, replay_size = r.number('q'), r.number('i')
            if replay_size != -1:
                if not 0 <= replay_size <= 25_000_000 or r.offset + replay_size > len(r.data):
                    raise ValueError('Некорректная длина реплея в scores.db.')
                r.offset += replay_size
            online_id = r.number('q')
            if mods & (1 << 23):
                r.number('d')  # Target Practice payload.
            try:
                played_at = (datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat()
            except (OverflowError, ValueError):
                continue
            rows.append({'mode': mode, 'checksum': checksum or group_hash, 'user': user,
                         'replay_hash': replay_hash, 'counts': counts, 'mods_bits': mods,
                         'online_id': online_id, 'played_at': played_at, 'score': score,
                         'combo': combo, 'client_version': client_version})
    return rows


def _config(root: Path):
    result = {}
    for path in root.glob('osu!.*.cfg'):
        if path.stat().st_size > 1_000_000:
            continue
        for line in path.read_text(encoding='utf-8-sig', errors='replace').splitlines():
            key, sep, value = line.partition('=')
            if sep and key.strip() in ('Username', 'BeatmapDirectory'):
                result[key.strip()] = value.strip()
    return result


def discover_clients(extra_path: str | None = None):
    candidates = []
    if extra_path:
        candidates.append(Path(extra_path).expanduser())
    if os.getenv('OSU_PATH'):
        candidates.append(Path(os.environ['OSU_PATH']))
    for variable, suffix in [('LOCALAPPDATA', 'osu!'), ('APPDATA', 'osu'), ('ProgramFiles', 'osu!')]:
        if os.getenv(variable):
            candidates.append(Path(os.environ[variable]) / suffix)
    candidates += [Path.home() / 'osu!', Path.home() / '.local/share/osu', Path.home() / 'Library/Application Support/osu']
    try:
        import psutil
        for process in psutil.process_iter(['name', 'exe']):
            if (process.info['name'] or '').lower() in ('osu!.exe', 'osu.exe', 'osu!') and process.info['exe']:
                candidates.append(Path(process.info['exe']).parent)
    except (ImportError, OSError):
        pass
    if os.name == 'nt':
        import winreg
        for key_name in (r'osu\shell\open\command', r'osu!\shell\open\command', r'Applications\osu!.exe\shell\open\command'):
            try:
                with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, key_name) as key:
                    command = winreg.QueryValue(key, None)
                match = re.match(r'"([^"]+\.exe)"|(.+?\.exe)', command, re.I)
                if match:
                    candidates.append(Path(match.group(1) or match.group(2)).parent)
            except OSError:
                pass
    found, seen = [], set()
    for candidate in candidates:
        try:
            root = candidate.resolve()
            if root.is_file():
                root = root.parent
            if str(root).casefold() in seen or not root.is_dir():
                continue
            seen.add(str(root).casefold())
            pointer = root / 'storage.ini'
            if pointer.is_file() and pointer.stat().st_size < 16000:
                for line in pointer.read_text(encoding='utf-8-sig', errors='replace').splitlines():
                    name, separator, target = line.partition('=')
                    if separator and name.strip() == 'FullPath' and target.strip():
                        relocated = Path(target.strip()).expanduser()
                        if not relocated.is_absolute():relocated = root / relocated
                        if relocated.resolve() != root:
                            candidates.append(relocated)
                            break
                else:
                    relocated = None
                if relocated is not None and relocated.resolve() != root:
                    continue
            stable, lazer = (root / 'scores.db').is_file(), (root / 'client.realm').is_file()
            if stable or lazer:
                found.append({'path': str(root), 'client': 'stable' if stable else 'lazer',
                              'supported': True, 'username': _config(root).get('Username') if stable else None})
        except OSError:
            continue
    return found


def _song_files(root: Path):
    directory = _config(root).get('BeatmapDirectory', 'Songs')
    songs = Path(directory)
    if not songs.is_absolute():
        songs = root / songs
    scanned = 0
    for current, dirs, names in os.walk(songs, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(current) / d).is_symlink()]
        for name in names:
            if name.lower().endswith('.osu'):
                scanned += 1
                if scanned > 200_000:
                    return
                yield Path(current) / name


def import_local_scores(store: Store, path: str | None = None, user: str | None = None, force=False):
    installations = discover_clients(path)
    if path:
        requested = Path(path).expanduser().resolve()
        if requested.is_file():
            requested = requested.parent
        visited = set()
        while requested not in visited:
            visited.add(requested)
            pointer = requested / 'storage.ini'
            if not pointer.is_file() or pointer.stat().st_size >= 16000:
                break
            targets = [line.partition('=')[2].strip() for line in pointer.read_text(encoding='utf-8-sig').splitlines()
                       if line.partition('=')[0].strip() == 'FullPath' and line.partition('=')[2].strip()]
            if not targets:
                break
            target = Path(targets[0]).expanduser()
            requested = (target if target.is_absolute() else requested / target).resolve()
        installations = [item for item in installations if Path(item['path']) == requested]
    stable = next((item for item in installations if item['supported']), None)
    if not stable:
        raise ValueError('osu! не найден. Укажите папку stable с scores.db или папку lazer с client.realm.')
    root = Path(stable['path'])
    if stable['client'] == 'lazer':
        from .lazer_client import import_lazer_scores
        return import_lazer_scores(store, root, user, force)
    db_path = root / 'scores.db'
    stat = db_path.stat()
    if stat.st_size > MAX_DB_BYTES:
        raise ValueError('scores.db превышает 256 МБ.')
    cache_key = f"local-v2:{root}:{(user or stable['username'] or '').casefold()}:{stat.st_mtime_ns}:{stat.st_size}"
    if not force:
        cached = store.cached_import(cache_key, ttl=None)
        if cached:
            return {**cached, 'imported': 0, 'cached': True}
    raw = db_path.read_bytes()
    after = db_path.stat()
    if (stat.st_mtime_ns, stat.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError('Клиент изменил scores.db во время чтения. Повторите импорт.')
    rows = read_scores(raw)
    players = Counter(row['user'] for row in rows if row['mode'] == 0 and row['user'])
    requested = user or stable['username'] or (players.most_common(1)[0][0] if players else '')
    if not requested:
        raise ValueError('В scores.db нет результатов osu!standard.')
    selected = [row for row in rows if row['mode'] == 0 and row['user'].casefold() == requested.casefold()]
    if not selected:
        raise ValueError('В этом клиенте нет результатов выбранного игрока. Выберите локальное имя игрока.')
    needed = {row['checksum'] for row in selected}
    maps = {}
    for checksum in needed:
        existing = store.get_map_by_checksum(checksum)
        if existing:
            maps[checksum] = existing
    if needed - maps.keys():
        for filename in _song_files(root):
            try:
                info = filename.stat()
                if info.st_size > 8_000_000:
                    continue
                checksum = store.cached_file_hash(filename, info.st_mtime_ns, info.st_size)
                data = None
                if checksum is None:
                    data = filename.read_bytes()
                    checksum = hashlib.md5(data).hexdigest()
                    store.cache_file_hash(filename, info.st_mtime_ns, info.st_size, checksum)
                if checksum in needed and checksum not in maps:
                    data = data if data is not None else filename.read_bytes()
                    if hashlib.md5(data).hexdigest() != checksum:
                        continue
                    parsed = parse_map(data)
                    previous = store.get_map(parsed.map_id)
                    if previous and previous.checksum != checksum:
                        parsed = parse_map(data, f'local-{checksum}')
                    store.save_map(parsed)
                    maps[checksum] = parsed
                if not needed - maps.keys():
                    break
            except (OSError, ValueError):
                continue
    imported = skipped = incomplete = 0
    allowed_names = ('NF', 'EZ', 'HD', 'HR', 'DT', 'HT', 'NC', 'FL')
    allowed_bits = sum(MOD_BITS[name] for name in allowed_names)
    for row in selected:
        beatmap = maps.get(row['checksum'])
        if not beatmap:
            skipped += 1
            continue
        bits = row['mods_bits']
        if bits & ~allowed_bits:
            skipped += 1
            continue
        n300, n100, n50, _, _, misses = row['counts']
        total = n300 + n100 + n50 + misses
        if not total or total != beatmap.metadata['object_count']:
            incomplete += 1
            continue
        mods, _ = normalize_mods([name for name in allowed_names if bits & MOD_BITS[name]] + ['CL'])
        accuracy = (300 * n300 + 100 * n100 + 50 * n50) / (300 * total)
        identity = f"osu-{row['online_id']}" if row['online_id'] > 0 else 'local-' + hashlib.sha256(
            f"{requested.casefold()}:{row['checksum']}:{mods}:{row['played_at']}:{row['score']}:{row['replay_hash']}".encode()).hexdigest()
        imported += int(store.save_play(requested, beatmap.map_id, mods, accuracy, False, misses,
                                       row['played_at'], 'local_unknown', identity, passed_known=False))
    replay_files = sum(1 for directory in (root / 'Replays', root / 'Data/r') if directory.is_dir()
                       for _ in directory.glob('*.osr'))
    result = {'username': requested, 'provider': 'local_stable', 'imported': imported,
              'total_plays': store.play_count(requested), 'local_scores': len(selected), 'skipped': skipped,
              'missing_maps': len(needed - maps.keys()), 'incomplete_scores': incomplete,
              'available_replays': replay_files, 'local_players': list(players), 'cached': False,
              'highlight_map_id': next((m.map_id for m in maps.values()), None),
              'message': 'Исход прохождения в scores.db неизвестен: локальные результаты не обучают pass. Импортированы полные результаты для accuracy.'}
    store.cache_import(cache_key, result)
    return result
