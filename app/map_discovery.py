"""Bounded discovery of unseen charts; download only .osu files, never audio."""
import asyncio

import httpx

from .analyzer import normalize_mods, parse_map
from .osu_api import download_beatmap, USER_AGENT
from .predictor import recommendation_bounds


async def discover_maps(store, user, goal='comfort', mods='NM'):
    mods, _ = normalize_mods(mods)
    lower, upper = recommendation_bounds(store, user, goal)
    key = f'discovery-v1:{user.casefold()}:{goal}:{mods}:{lower}:{upper}'
    cached = store.cached_import(key, ttl=600)
    if cached:
        return {**cached, 'cached': True}
    excluded = store.excluded_map_ids(user)
    candidates = []
    async with httpx.AsyncClient(timeout=20, headers={'User-Agent': USER_AGENT}) as client:
        for offset in (0, 30, 60):
            response = await client.get('https://mirror.hinamizawa.ai/api/v1/hinai/search', params={
                'mode': 0, 'status': 1, 'amount': 30, 'offset': offset, 'sort': 'ranked_desc',
                # Mirror metadata is NM. Final limits are checked with actual selected mods.
                'min_stars': lower if mods in ('NM', 'HD', 'CL', 'HDCL') else lower * .5,
                'max_stars': upper if 'HT' not in mods and 'EZ' not in mods else upper * 2,
            })
            response.raise_for_status()
            sets = response.json()
            if not isinstance(sets, list):
                raise ValueError('Источник карт вернул неожиданный формат.')
            for beatmapset in sets:
                for row in beatmapset.get('ChildrenBeatmaps', []):
                    map_id = str(row.get('BeatmapID', ''))
                    stars = float(row.get('DifficultyRating') or 0)
                    if row.get('Mode') != 0 or not map_id.isdigit() or map_id in excluded or map_id in candidates:
                        continue
                    if mods in ('NM', 'HD', 'CL', 'HDCL') and not lower <= stars <= upper:
                        continue
                    candidates.append(map_id)
                    if len(candidates) >= 24:
                        break
                if len(candidates) >= 24:
                    break
            if len(candidates) >= 24 or len(sets) < 30:
                break
    semaphore = asyncio.Semaphore(3)

    async def add(map_id):
        async with semaphore:
            try:
                if not store.get_map(map_id):
                    store.save_map(parse_map(await download_beatmap(map_id), map_id))
                    await asyncio.sleep(.15)
                analysis = store.analysis(map_id, mods)
                return int(bool(analysis and lower <= analysis['stars'] <= upper))
            except (httpx.HTTPError, ValueError, RuntimeError):
                return 0
    available = sum(await asyncio.gather(*(add(m) for m in candidates)))
    result = {'candidates': len(candidates), 'available': available, 'cached': False,
              'message': 'Новые карты найдены в открытом каталоге Hinamizawa. Знакомые и слишком простые карты исключены по доступной истории.'}
    store.cache_import(key, result)
    return result
