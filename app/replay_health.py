"""Run the official osu! replay player and HP processor in an isolated process."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from functools import lru_cache
from datetime import datetime, timedelta, timezone

from .storage import Store

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'tools/HealthReplayEngine/HealthReplayEngine.csproj'
ENGINE_VERSION = 'osu-2026.916.0-hp-v1'
_RUN_LOCK = threading.Lock()


def _engine():
    local = ROOT / '.data/tools/dotnet/dotnet.exe'
    dotnet = str(local) if local.exists() else shutil.which('dotnet')
    if not dotnet:
        raise ValueError('Для расчета HP нужен .NET SDK 10. Установите его и повторите расчет.')
    binary = PROJECT.parent / 'bin/Release/net10.0/HealthReplayEngine.dll'
    newest = max(p.stat().st_mtime for p in PROJECT.parent.glob('*') if p.suffix in ('.cs', '.csproj'))
    if not binary.exists() or binary.stat().st_mtime < newest:
        result = subprocess.run([dotnet, 'build', str(PROJECT), '-c', 'Release', '--nologo', '-v:q'],
                                capture_output=True, timeout=240,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        if result.returncode:
            raise ValueError('Не удалось собрать обработчик HP. Нужен .NET SDK 10 и доступ к NuGet.')
    _verify_silent_engine(dotnet, str(binary), binary.stat().st_mtime_ns)
    return dotnet, binary


@lru_cache(maxsize=8)
def _verify_silent_engine(dotnet, binary, stamp):
    """Fail closed before loading any replay if native audio guards do not work."""
    base = ROOT / '.data/audio-checks'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='check-', dir=base) as temp:
        output = Path(temp) / 'audio.json'
        try:
            result = subprocess.run([dotnet, binary, '--audio-check', str(output)], cwd=temp,
                capture_output=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            check = json.loads(output.read_text(encoding='utf-8-sig')) if output.is_file() else {}
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            raise ValueError('Не удалось проверить отключение звука. Обработка реплеев не запущена.') from exc
    if (result.returncode or check.get('status') != 'verified' or not check.get('audio_muted')
            or check.get('device') != 'No sound' or not check.get('no_replay_loaded')
            or any(check.get(k) != 0 for k in ('universal_volume', 'sample_volume', 'track_volume'))):
        raise ValueError('Защита от звука не прошла проверку. Обработка реплеев не запущена.')
    return True


def calculate_health(store: Store, user: str, play_id: str):
    play = store.replay_input(user, play_id)
    if not play or 'NF' not in play['mods']:
        raise ValueError('Не найден локальный NF-реплей этого игрока. Повторите импорт клиента.')
    path = Path(play['path'])
    if not path.is_file() or path.stat().st_size > 25_000_000:
        raise ValueError('Реплей отсутствует или превышает 25 МБ. Повторите локальный импорт.')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != play['sha256']:
        raise ValueError('Файл реплея изменился. Повторите импорт клиента.')
    beatmap = store.get_map(play['map_id'])
    if not beatmap:
        raise ValueError('Не найдена точная версия карты.')
    key = f'{ENGINE_VERSION}:{digest}:{beatmap.checksum}'
    with _RUN_LOCK:
        cached = store.health_cache(key)
        if cached:
            return {**cached, 'cached': True}
        dotnet, binary = _engine()
        base = ROOT / '.data/hp-runs'
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='run-', dir=base) as temp:
            folder = Path(temp)
            # Use copies only; the osu! client store is never passed to the engine.
            chart_path, replay_path, output = folder / 'map.osu', folder / 'replay.osr', folder / 'result.json'
            chart_bytes = beatmap.content.encode('utf-8')
            if hashlib.md5(chart_bytes).hexdigest() != beatmap.checksum:
                chart_bytes = b'\xef\xbb\xbf' + chart_bytes
            if hashlib.md5(chart_bytes).hexdigest() != beatmap.checksum:
                raise ValueError('Не удалось восстановить точные байты карты. Загрузите исходный .osu заново.')
            chart_path.write_bytes(chart_bytes)
            replay_path.write_bytes(raw)
            try:
                process = subprocess.run([dotnet, str(binary), str(chart_path), str(replay_path), str(output)],
                                         cwd=folder, capture_output=True, timeout=180,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                         env={**os.environ, 'DOTNET_CLI_TELEMETRY_OPTOUT': '1'})
            except subprocess.TimeoutExpired as exc:
                raise ValueError('Расчет HP превысил 3 минуты. Результат не сохранен.') from exc
            if process.returncode or not output.exists():
                raise ValueError('Движок не смог воспроизвести реплей. Проверьте его и версию карты в osu!.')
            result = json.loads(output.read_text(encoding='utf-8-sig'))
        if result.get('status') not in ('verified', 'statistics_mismatch'):
            raise ValueError('Воспроизведение реплея завершилось ошибкой.')
        result['cached'] = False
        result['play_id'] = play_id
        result['complete_score'] = bool(play.get('complete', 1))
        result['verified_counterfactual'] = (result['status'] == 'verified' and bool(play.get('complete', 1))
                                             and 'EZ' not in result.get('mods', []))
        result['message'] = ('HP рассчитан обработчиком osu!. Статистика воспроизведения совпала с сохраненной. Это расчет по версии движка, а не новая сыгранная попытка.'
                             if result['verified_counterfactual'] else 'Статистика отличается, попытка неполная или есть EZ с дополнительными жизнями. График HP показан, но конечный исход без NF не подтвержден.')
        store.save_health_cache(key, result)
        return result


def register_replay(store: Store, user: str, data: bytes):
    from .local_client import Reader
    from .analyzer import MOD_BITS, normalize_mods
    if not user.strip() or len(user) > 50 or not data or len(data) > 25_000_000:
        raise ValueError('Укажите игрока и реплей не больше 25 МБ.')
    try:
        r = Reader(data)
        mode, version = r.number('B'), r.number('i')
        checksum, owner, _ = r.string(), r.string(), r.string()
        counts = [r.number('H') for _ in range(6)]
        r.number('i'); r.number('H'); r.number('B')
        bits = r.number('i')
        r.string()  # HP graph is optional and not used to infer the result.
        ticks = r.number('q')
        played_at = (datetime(1, 1, 1, tzinfo=timezone.utc) + timedelta(microseconds=ticks // 10)).isoformat()
    except (ValueError, UnicodeError, OverflowError) as exc:
        raise ValueError('Не удалось прочитать заголовок .osr.') from exc
    if mode != 0 or not bits & MOD_BITS['NF']:
        raise ValueError('Нужен реплей osu!standard с No Fail.')
    if owner.strip().casefold() != user.strip().casefold():
        raise ValueError('Реплей принадлежит другому игроку. Выберите его профиль.')
    beatmap = store.get_map_by_checksum(checksum)
    if not beatmap:
        raise ValueError('Сначала загрузите точную карту .osu из этого реплея.')
    digest = hashlib.sha256(data).hexdigest()
    with store._connection() as db:
        existing = db.execute('''SELECT p.id FROM plays p JOIN local_replays r ON p.id=r.play_id
                                WHERE p.user_key=? AND r.sha256=?''', (user.strip().casefold(), digest)).fetchone()
    if existing:
        return existing['id']
    folder = ROOT / '.data/replays'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{digest}.osr'
    path.write_bytes(data)
    mods = [name for name, bit in MOD_BITS.items() if bit and bits & bit]
    if version < 30_000_000:
        mods.append('CL')
    label, _ = normalize_mods(mods)
    total = sum(counts[i] for i in (0, 1, 2, 5))
    accuracy = (300 * counts[0] + 100 * counts[1] + 50 * counts[2]) / max(1, 300 * total)
    identity = f'nf-analysis-{digest}'
    store.save_play(user, beatmap.map_id, label, accuracy, False, counts[5], played_at=played_at, source='replay_nf_analysis',
                    play_id=identity, passed_known=False, complete=total == beatmap.metadata['object_count'])
    store.save_local_replay(identity, path, digest)
    return identity
