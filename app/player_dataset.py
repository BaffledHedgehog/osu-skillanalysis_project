"""Collect an explicitly selected player's available history without replay playback."""
from __future__ import annotations

import asyncio
from pathlib import Path
import re

import httpx

from .local_client import discover_clients, import_local_scores, _song_files
from .osu_api import import_user_scores, parse_player_reference


def collect_owned_maps(store, client, user):
    root = Path(client['path'])
    if client['client'] == 'lazer':
        from .lazer_client import _read_realm, stored_file
        payload = _read_realm(root, include_inventory=True)
        ids = {str(row['id']) for row in payload['beatmaps'] if row.get('id', 0) > 0
               and stored_file(root, row.get('hash'))}
    else:
        ids = set()
        for file in _song_files(root):
            try:
                with file.open('rb') as f:
                    header = f.read(16384).decode('utf-8-sig', errors='replace').split('[HitObjects]', 1)[0]
                match = re.search(r'^BeatmapID\s*:\s*(\d+)\s*$', header, re.MULTILINE)
                if match and int(match[1]) > 0:
                    ids.add(match[1])
            except OSError:
                continue
    store.replace_owned_maps(user, str(root.resolve()), ids)
    return len(ids)


async def collect_dataset(store, reference, *, allow_local, force=False, cookies=None, oauth=None, progress=None):
    report = progress or (lambda **kwargs: None)
    reference = parse_player_reference(reference)
    report(phase='online', message='Загружаем recent, best, pinned и firsts, где они доступны.')
    sources, warnings = [], []
    online = None
    user = reference
    try:
        online = await import_user_scores(store, reference, force=force, max_scores=1000,
                                          cookies=cookies, oauth=oauth, expanded=True)
        user = online['username']
        sources.append(online)
    except (httpx.HTTPError, ValueError):
        warnings.append('Онлайн-источник не ответил или не принял доступ. Проверьте профиль и подключение в дополнительных настройках.')
        try:
            online = await import_user_scores(store, reference, force=force, max_scores=1000, expanded=True, use_public=True)
            user = online['username']
            sources.append(online)
            warnings.append('Для онлайн-истории использован открытый источник.')
        except (httpx.HTTPError, ValueError):
            pass
        if not online and reference.isdigit():
            # Never treat a numeric account ID as a local player name.
            raise ValueError('Не удалось определить имя игрока по ID. Повторите позже или укажите ник.')
    if allow_local:
        report(phase='local', message='Проверяем все найденные клиенты stable и lazer для этого игрока.')
        clients = await asyncio.to_thread(discover_clients)
        visited = set()
        for client in clients:
            root = str(Path(client['path']).resolve())
            if not client.get('supported') or root.casefold() in visited:
                continue
            visited.add(root.casefold())
            try:
                result = await asyncio.to_thread(import_local_scores, store, root, user, force)
            except (OSError, ValueError, UnicodeError):
                warnings.append(f"В одном из клиентов {client['client']} не удалось прочитать результаты выбранного игрока.")
                continue
            try:
                result['owned_maps'] = await asyncio.to_thread(collect_owned_maps, store, client, user)
            except (OSError, ValueError):
                result['inventory_unavailable'] = True
                warnings.append(f"Список установленных карт {client['client']} прочитан не полностью.")
            sources.append(result)
    else:
        warnings.append('Локальные файлы доступны при запуске приложения на компьютере игрока.')
    if not sources:
        raise ValueError('Не удалось собрать данные игрока. Проверьте ник, интернет или настройки локального клиента.')
    report(phase='complete', message='История дополнена. Повторные попытки не дублируются.')
    return {'username': user, 'sources': sources, 'warnings': warnings,
            'imported': sum(s.get('imported', 0) for s in sources), 'total_plays': store.play_count(user),
            'retained_scores': len(store.score_payloads(user)),
            'highlight_map_id': next((s.get('highlight_map_id') for s in sources if s.get('highlight_map_id')), None),
            'limits': 'Собраны доступные списки и локальные файлы этого игрока. Сайт не хранит в recent весь архив. Реплеи не проигрываются; HP и тяжелое обучение запускаются отдельно.'}
