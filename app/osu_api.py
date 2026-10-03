"""Polite, opt-in access to official osu! endpoints for one requested player."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx
from dotenv import load_dotenv

from .analyzer import normalize_mods, parse_map
from .storage import Store


USER_AGENT = "osu-skillanalysis/0.2 (Innopolis student project; local use)"
load_dotenv()

SCORES_PER_SOURCE = 100
MAX_SCORE_PAGES = 10
PUBLIC_API = "https://mirror.hinamizawa.ai/api/v1/hinai/player"
PROFILE_PATH = re.compile(r"^/users/(\d{1,12})(?:/(?:osu|taiko|fruits|mania))?/?$")


def parse_player_reference(value: str) -> str:
    """Accept a username, numeric user ID, or an official osu! profile URL."""
    value = value.strip()
    if not value or len(value) > 300:
        raise ValueError("Укажите ник, ID или ссылку на профиль osu!.")
    if value.startswith(("osu.ppy.sh/", "www.osu.ppy.sh/")):
        value = "https://" + value
    if "://" in value:
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in ("osu.ppy.sh", "www.osu.ppy.sh"):
            raise ValueError("Нужна ссылка на профиль с osu.ppy.sh/users/ID.")
        match = PROFILE_PATH.fullmatch(parsed.path)
        if not match:
            raise ValueError("Нужна ссылка на профиль с osu.ppy.sh/users/ID.")
        return match.group(1)
    if "/" in value or "\\" in value or "?" in value or "#" in value:
        raise ValueError("Укажите ник, ID или ссылку на профиль osu!.")
    value = unquote(value)
    if not value or len(value) > 50 or "/" in value or "\\" in value:
        raise ValueError("Укажите корректное имя игрока.")
    return value


async def download_beatmap(beatmap_id: str) -> bytes:
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        response = await client.get(f"https://osu.ppy.sh/osu/{beatmap_id}")
        response.raise_for_status()
        if len(response.content) > 8_000_000:
            raise ValueError("Файл карты превышает лимит 8 МБ.")
        return response.content


def api_configured() -> bool:
    return bool(os.getenv("OSU_CLIENT_ID") and os.getenv("OSU_CLIENT_SECRET"))


async def _token(client: httpx.AsyncClient) -> str:
    response = await client.post("https://osu.ppy.sh/oauth/token", json={
        "client_id": os.environ["OSU_CLIENT_ID"],
        "client_secret": os.environ["OSU_CLIENT_SECRET"],
        "grant_type": "client_credentials", "scope": "public",
    })
    response.raise_for_status()
    return response.json()["access_token"]


async def import_user_scores(store: Store, username: str) -> dict[str, Any]:
    reference = parse_player_reference(username)
    provider = "osu_api" if api_configured() else "hinai"
    selected: dict[str, dict[str, Any]] = {}
    selected_by_source = {"recent": 0, "best": 0}
    received_by_source = {"recent": 0, "best": 0}
    skipped = 0
    duplicates = 0
    page_limit_hit: list[str] = []
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        headers = {"Accept": "application/json"}
        if provider == "osu_api":
            token = await _token(client)
            headers["Authorization"] = f"Bearer {token}"
            profile_url = ("https://osu.ppy.sh/api/v2/users/"
                           f"{reference if reference.isdigit() else '@' + quote(reference, safe='')}/osu")
        else:
            profile_url = f"{PUBLIC_API}/{quote(reference, safe='')}"
        response = await client.get(profile_url, headers=headers)
        response.raise_for_status()
        player = response.json()
        user_id = player["id"]
        canonical_username = player["username"]
        # Recent is processed first, so a recent fail is retained if its map
        # also appears among best scores. Later pages fill repeated chart IDs.
        for source in ("recent", "best"):
            offset = 0
            previous_page_ids: tuple[str, ...] | None = None
            page_count = MAX_SCORE_PAGES if provider == "osu_api" else 1
            for page in range(page_count):
                if selected_by_source[source] >= SCORES_PER_SOURCE:
                    break
                if provider == "osu_api":
                    score_url = f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/{source}"
                    params = {"mode": "osu", "limit": SCORES_PER_SOURCE, "offset": offset}
                    if source == "recent":
                        params["include_fails"] = 1
                else:
                    score_url = f"{PUBLIC_API}/{user_id}/scores"
                    params = {"type": source, "mode": "osu", "limit": SCORES_PER_SOURCE}
                scores_response = await client.get(score_url, headers=headers, params=params)
                # Some score lists expose only the first 100 rows. A later page
                # may be rejected instead of returning an empty list.
                if provider == "osu_api" and offset and scores_response.status_code in (400, 404, 416, 422):
                    break
                scores_response.raise_for_status()
                payload = scores_response.json()
                scores = payload.get("scores") if provider == "hinai" and isinstance(payload, dict) else payload
                if not isinstance(scores, list):
                    raise ValueError("Источник данных вернул неожиданный формат результатов.")
                page_ids = tuple(str(score.get("id")) for score in scores)
                if page_ids == previous_page_ids:
                    break
                previous_page_ids = page_ids
                received_by_source[source] += len(scores)
                for score in scores:
                    beatmap = score.get("beatmap") or {}
                    map_id = str(beatmap.get("id") or score.get("beatmap_id") or "")
                    if not map_id.isdigit():
                        skipped += 1
                        continue
                    raw_mods = score.get("mods") or []
                    if any(isinstance(mod, dict) and mod.get("settings") for mod in raw_mods):
                        skipped += 1
                        continue
                    acronyms = [(mod.get("acronym", "") if isinstance(mod, dict) else str(mod)).upper() for mod in raw_mods]
                    try:
                        mods, _ = normalize_mods(acronyms)
                    except ValueError:
                        skipped += 1
                        continue
                    if "NF" in acronyms:
                        skipped += 1
                        continue
                    accuracy = score.get("accuracy")
                    if accuracy is None or score.get("passed") is None or score.get("id") is None:
                        skipped += 1
                        continue
                    try:
                        accuracy = float(accuracy)
                    except (TypeError, ValueError):
                        skipped += 1
                        continue
                    if 1 < accuracy <= 100:
                        accuracy /= 100
                    if not 0 <= accuracy <= 1:
                        skipped += 1
                        continue
                    if map_id in selected:
                        duplicates += 1
                        continue
                    selected[map_id] = {"score": score, "mods": mods, "source": source,
                                        "accuracy": accuracy}
                    selected_by_source[source] += 1
                    if selected_by_source[source] >= SCORES_PER_SOURCE:
                        break
                offset += len(scores)
                if len(scores) < SCORES_PER_SOURCE:
                    break
            else:
                if provider == "osu_api" and selected_by_source[source] < SCORES_PER_SOURCE:
                    page_limit_hit.append(source)

    fetched = 0
    imported = 0
    # A single explicit profile import can contain up to 200 different charts.
    # Limit concurrent file requests to keep the public chart endpoint responsive.
    semaphore = asyncio.Semaphore(3)

    async def ensure_map(map_id: str) -> bool:
        nonlocal fetched
        if store.get_map(map_id):
            return True
        async with semaphore:
            try:
                raw = await download_beatmap(map_id)
                store.save_map(parse_map(raw, map_id))
                fetched += 1
                await asyncio.sleep(0.15)
                return True
            except (httpx.HTTPError, ValueError):
                return False

    map_ids = list(selected)
    availability = await asyncio.gather(*(ensure_map(map_id) for map_id in map_ids))
    highlight_map_id = next((map_id for map_id, available in zip(map_ids, availability) if available), None)
    for map_id, available in zip(map_ids, availability):
        if not available:
            skipped += 1
            continue
        score = selected[map_id]["score"]
        mods = selected[map_id]["mods"]
        source = selected[map_id]["source"]
        try:
            stats = score.get("statistics") or {}
            raw_pp = score.get("pp")
            pp = float(raw_pp) if raw_pp is not None else None
            changed = store.save_play(
                canonical_username, map_id, mods, selected[map_id]["accuracy"], bool(score["passed"]),
                int(stats.get("miss", stats.get("count_miss", 0)) or 0),
                score.get("ended_at") or score.get("created_at"),
                f"{provider}_{source}", f"osu-{score['id']}", pp,
            )
            imported += int(changed)
        except (ValueError, KeyError, TypeError):
            skipped += 1
    return {"username": canonical_username, "user_id": user_id, "provider": provider,
            "imported": imported,
            "skipped": skipped, "duplicates": duplicates, "maps_fetched": fetched,
            "received_recent": received_by_source["recent"],
            "received_best": received_by_source["best"],
            "unique_maps": len(selected), "recent_unique": selected_by_source["recent"],
            "best_unique": selected_by_source["best"], "highlight_map_id": highlight_map_id,
            "page_limit_hit": page_limit_hit,
            "total_plays": store.play_count(canonical_username)}
