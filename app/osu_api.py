"""Polite, opt-in access to official osu! endpoints for one requested player."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

from .analyzer import normalize_mods, parse_map
from .storage import Store


USER_AGENT = "osu-skillanalysis/0.1 (personal local app)"
load_dotenv()


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
    if not api_configured():
        raise ValueError("Для импорта добавьте OSU_CLIENT_ID и OSU_CLIENT_SECRET в окружение сервера.")
    username = username.strip()
    if not username or len(username) > 50:
        raise ValueError("Укажите корректное имя игрока.")
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        token = await _token(client)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        response = await client.get(
            f"https://osu.ppy.sh/api/v2/users/@{quote(username, safe='')}/osu", headers=headers
        )
        response.raise_for_status()
        player = response.json()
        user_id = player["id"]
        best_response, recent_response = await asyncio.gather(
            client.get(f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/best",
                       headers=headers, params={"mode": "osu", "limit": 30}),
            client.get(f"https://osu.ppy.sh/api/v2/users/{user_id}/scores/recent",
                       headers=headers, params={"mode": "osu", "limit": 30, "include_fails": 1}),
        )
        best_response.raise_for_status()
        recent_response.raise_for_status()
        scores = best_response.json() + recent_response.json()

    imported = 0
    skipped = 0
    fetched = 0
    # Do not flood the beatmap endpoint: at most 40 new chart files per explicit import.
    for score in scores:
        beatmap = score.get("beatmap") or {}
        map_id = str(beatmap.get("id") or score.get("beatmap_id") or "")
        if not map_id.isdigit():
            skipped += 1
            continue
        raw_mods = score.get("mods") or []
        if any(isinstance(mod, dict) and mod.get("settings") for mod in raw_mods):
            # Custom rates and difficulty settings cannot be represented by legacy mod bits.
            skipped += 1
            continue
        acronyms = [mod.get("acronym", "") if isinstance(mod, dict) else str(mod) for mod in raw_mods]
        try:
            mods, _ = normalize_mods(acronyms)
        except ValueError:
            skipped += 1
            continue
        if "NF" in acronyms:
            # No Fail changes the meaning of a successful pass.
            skipped += 1
            continue
        if not store.get_map(map_id):
            if fetched >= 40:
                skipped += 1
                continue
            try:
                raw = await download_beatmap(map_id)
                store.save_map(parse_map(raw, map_id))
                fetched += 1
                await asyncio.sleep(0.15)
            except (httpx.HTTPError, ValueError):
                skipped += 1
                continue
        accuracy = score.get("accuracy")
        if accuracy is None:
            skipped += 1
            continue
        try:
            changed = store.save_play(
                username, map_id, mods, float(accuracy), bool(score.get("passed", True)),
                int((score.get("statistics") or {}).get("miss") or 0),
                score.get("created_at"), "osu_api", f"osu-{score['id']}",
            )
            imported += int(changed)
        except (ValueError, KeyError, TypeError):
            skipped += 1
    return {"username": player.get("username", username), "imported": imported,
            "skipped": skipped, "maps_fetched": fetched, "total_plays": store.play_count(username)}
