"""Parse osu!standard beatmaps and calculate official-style difficulty attributes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import rosu_pp_py


MOD_BITS = {"NF": 1, "EZ": 2, "HD": 8, "HR": 16, "DT": 64, "HT": 256, "NC": 512, "FL": 1024}
VISIBLE_MODS = ("NM", "HD", "HR", "DT", "NC", "EZ", "HT", "FL", "NF")


def normalize_mods(value: str | list[str] | None) -> tuple[str, int]:
    if isinstance(value, list):
        tokens = [str(token).upper() for token in value]
    else:
        raw = (value or "NM").upper().replace("+", "").replace(" ", "")
        tokens = [raw[i : i + 2] for i in range(0, len(raw), 2)] if raw != "NM" else ["NM"]
    if not tokens or tokens == ["NM"]:
        return "NM", 0
    if any(token not in MOD_BITS for token in tokens):
        raise ValueError("Поддерживаются только NM, HD, HR, DT, NC, EZ, HT, FL и NF.")
    unique = set(tokens)
    if "DT" in unique and "NC" in unique:
        unique.remove("DT")
    if ("DT" in unique or "NC" in unique) and "HT" in unique:
        raise ValueError("DT/NC и HT нельзя использовать вместе.")
    if "HR" in unique and "EZ" in unique:
        raise ValueError("HR и EZ нельзя использовать вместе.")
    bits = sum(MOD_BITS[token] for token in unique)
    if "NC" in unique:
        bits |= MOD_BITS["DT"]
    ordered = [token for token in VISIBLE_MODS if token in unique]
    return "".join(ordered), bits


def _sections(content: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def _pairs(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def _as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ParsedMap:
    map_id: str
    checksum: str
    content: str
    metadata: dict[str, Any]


def parse_map(raw: bytes, forced_id: str | None = None) -> ParsedMap:
    if len(raw) > 8_000_000:
        raise ValueError("Файл .osu слишком большой (лимит 8 МБ).")
    content = raw.decode("utf-8-sig", errors="replace")
    if not content.startswith("osu file format"):
        raise ValueError("Это не файл карты .osu.")
    sections = _sections(content)
    general = _pairs(sections.get("General", []))
    if general.get("Mode", "0") != "0":
        raise ValueError("Пока поддерживаются только карты osu!standard.")
    meta = _pairs(sections.get("Metadata", []))
    diff = _pairs(sections.get("Difficulty", []))
    objects: list[tuple[float, float, float, int]] = []
    for line in sections.get("HitObjects", []):
        parts = line.split(",", 5)
        if len(parts) >= 4:
            try:
                objects.append((float(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])))
            except ValueError:
                continue
    if len(objects) < 2:
        raise ValueError("В карте недостаточно игровых объектов для анализа.")
    try:
        rosu_map = rosu_pp_py.Beatmap(content=raw)
        if rosu_map.mode != rosu_pp_py.GameMode.Osu or rosu_map.n_objects < 2:
            raise ValueError("Файл не содержит корректную карту osu!standard.")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Не удалось прочитать структуру карты .osu.") from exc
    checksum = hashlib.md5(raw).hexdigest()
    beatmap_id = forced_id or meta.get("BeatmapID", "")
    if not beatmap_id or beatmap_id in ("0", "-1"):
        beatmap_id = "local-" + checksum[:12]
    first, last = objects[0][2], objects[-1][2]
    duration = max(1.0, (last - first) / 1000)
    bins = max(1, math.ceil(duration / 10))
    density = [0] * bins
    for _, _, time, _ in objects:
        idx = min(bins - 1, max(0, int((time - first) / 10_000)))
        density[idx] += 1
    metadata = {
        "artist": meta.get("Artist", "Unknown artist"),
        "title": meta.get("Title", "Untitled"),
        "version": meta.get("Version", "Unknown"),
        "creator": meta.get("Creator", "Unknown"),
        "ar": _as_float(diff.get("ApproachRate"), _as_float(diff.get("OverallDifficulty"), 5)),
        "od": _as_float(diff.get("OverallDifficulty"), 5),
        "cs": _as_float(diff.get("CircleSize"), 5),
        "hp": _as_float(diff.get("HPDrainRate"), 5),
        "object_count": len(objects),
        "duration": round(duration),
        "density": density,
        "first_object_ms": int(first),
    }
    return ParsedMap(str(beatmap_id), checksum, content, metadata)


def _strain_segments(content: bytes, bits: int) -> list[dict[str, Any]]:
    beatmap = rosu_pp_py.Beatmap(content=content)
    strain = rosu_pp_py.Difficulty(mods=bits).strains(beatmap)
    aim = list(strain.aim or [])
    speed = list(strain.speed or [])
    step = float(strain.section_length or 400) / 1000
    # Three-second windows are legible on screen and reduce noise from one hit object.
    per_window = max(1, round(3 / step))
    windows = []
    for i in range(0, max(len(aim), len(speed)), per_window):
        av = max(aim[i : i + per_window], default=0)
        sv = max(speed[i : i + per_window], default=0)
        windows.append({"start": round(i * step, 1), "end": round((i + per_window) * step, 1),
                        "aim": round(av, 2), "speed": round(sv, 2), "load": round(av + sv, 2)})
    return windows


def analyze_map(parsed: ParsedMap, mods: str = "NM") -> dict[str, Any]:
    mods_label, bits = normalize_mods(mods)
    raw = parsed.content.encode("utf-8")
    beatmap = rosu_pp_py.Beatmap(content=raw)
    attrs = rosu_pp_py.Difficulty(mods=bits).calculate(beatmap)
    windows = _strain_segments(raw, bits)
    peaks = sorted(windows, key=lambda window: window["load"], reverse=True)[:3]
    speed_factor = 1.5 if bits & MOD_BITS["DT"] else 0.75 if bits & MOD_BITS["HT"] else 1.0
    return {
        "id": parsed.map_id,
        "checksum": parsed.checksum,
        **parsed.metadata,
        "mods": mods_label,
        "stars": round(float(attrs.stars), 2),
        "aim": round(float(attrs.aim or 0), 2),
        "speed": round(float(attrs.speed or 0), 2),
        "bpm": round(float(beatmap.bpm) * speed_factor),
        "duration": round(parsed.metadata["duration"] / speed_factor),
        "ar": round(float(attrs.ar or parsed.metadata["ar"]), 1),
        "od": round(float(parsed.metadata["od"]), 1),
        "max_combo": int(attrs.max_combo or 0),
        "windows": windows,
        "peaks": peaks,
    }
