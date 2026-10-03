"""Import summary results from exported osu!stable .osr replay files."""

from __future__ import annotations

import hashlib
from datetime import timezone
from typing import Any

from osrparse import GameMode, Replay

from .analyzer import MOD_BITS, normalize_mods
from .storage import Store


def import_replay(store: Store, raw: bytes, user: str, passed: bool) -> dict[str, Any]:
    if not raw or len(raw) > 25_000_000:
        raise ValueError("Файл реплея пустой или слишком большой (лимит 25 МБ).")
    try:
        replay = Replay.from_string(raw)
    except Exception as exc:
        raise ValueError("Не удалось прочитать .osr. Пока поддерживаются экспортированные реплеи osu!stable.") from exc
    if replay.mode != GameMode.STD:
        raise ValueError("Пока поддерживаются только реплеи osu!standard.")
    if replay.username.strip().casefold() != user.strip().casefold():
        raise ValueError(f"Реплей принадлежит игроку {replay.username}. Выберите этот профиль перед импортом.")
    beatmap = store.get_map_by_checksum(replay.beatmap_hash)
    if not beatmap:
        raise ValueError("Точная версия карты из реплея не найдена. Сначала загрузите соответствующий файл .osu или .osz.")
    bits = int(replay.mods)
    if bits & MOD_BITS["NF"]:
        raise ValueError("Реплеи с No Fail не подходят для обучения вероятности прохождения.")
    allowed = 0
    tokens = []
    for name in ("EZ", "HD", "HR", "DT", "HT", "NC", "FL"):
        allowed |= MOD_BITS[name]
        if bits & MOD_BITS[name]:
            tokens.append(name)
    if bits & ~allowed:
        raise ValueError("В реплее есть моды, которые эта версия приложения пока не поддерживает.")
    mods, _ = normalize_mods(tokens)
    n300, n100, n50, miss = replay.count_300, replay.count_100, replay.count_50, replay.count_miss
    total = n300 + n100 + n50 + miss
    if total < 1:
        raise ValueError("В реплее нет результатов попаданий.")
    accuracy = (300 * n300 + 100 * n100 + 50 * n50) / (300 * total)
    timestamp = replay.timestamp
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    play_id = "replay-" + (replay.replay_hash or hashlib.sha256(raw).hexdigest())
    created = store.save_play(user, beatmap.map_id, mods, accuracy, passed, miss,
                              timestamp.isoformat(), "replay", play_id)
    return {"created": created, "map_id": beatmap.map_id, "mods": mods,
            "accuracy": round(accuracy * 100, 2), "misses": miss, "passed": passed}
