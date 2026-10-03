"""Experimental personalized difficulty model, trained only on actual player results."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .ml import model_context, predict_pass, vector
from .storage import Store


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-_clamp(value, -25, 25)))


def _weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    ranked = sorted(zip(values, weights), key=lambda item: item[0])
    goal = q * sum(weights)
    total = 0.0
    for value, weight in ranked:
        total += weight
        if total >= goal:
            return value
    return ranked[-1][0]


def _records(store: Store, user: str) -> list[dict[str, Any]]:
    records = []
    now = datetime.now(timezone.utc)
    for play in store.plays(user):
        analysis = store.analysis(play["map_id"], play["mods"])
        if not analysis:
            continue
        try:
            played_at = datetime.fromisoformat(play["played_at"].replace("Z", "+00:00"))
            age_days = max(0, (now - played_at).total_seconds() / 86_400)
        except ValueError:
            age_days = 0
        play["analysis"] = analysis
        source_weight = 0.35 if play["source"] in ("osu_api_best", "hinai_best", "osu_api") else 1.0
        play["weight"] = source_weight * 0.5 ** (age_days / 180)
        records.append(play)
    return records


def _summary_from_records(user: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [r for r in records if r["passed"]]
    if not records:
        return {"user": user, "plays": 0, "passed": 0, "accuracy": None,
                "comfort_stars": None, "aim_level": None, "speed_level": None,
                "best_pp": None, "sources": {}, "recent": []}
    weights = [r["weight"] for r in records]
    source_counts: dict[str, int] = {}
    for r in records:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1
    return {
        "user": user,
        "plays": len(records),
        "passed": len(passed),
        "accuracy": round(sum(r["accuracy"] * r["weight"] for r in records) / sum(weights) * 100, 1),
        "best_pp": round(max((r["pp"] for r in records if r["pp"] is not None), default=0), 1)
                   if any(r["pp"] is not None for r in records) else None,
        "comfort_stars": round(_weighted_quantile(
            [r["analysis"]["stars"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "aim_level": round(_weighted_quantile(
            [r["analysis"]["aim"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "speed_level": round(_weighted_quantile(
            [r["analysis"]["speed"] for r in passed], [r["weight"] for r in passed], 0.7), 2) if passed else None,
        "sources": source_counts,
        "recent": [{"map_id": r["map_id"], "title": r["analysis"]["title"], "mods": r["mods"],
                    "accuracy": round(r["accuracy"] * 100, 1), "passed": bool(r["passed"])} for r in records[:5]],
    }


def user_summary(store: Store, user: str) -> dict[str, Any]:
    return _summary_from_records(user, _records(store, user))


def predict(store: Store, user: str, target: dict[str, Any],
            records: list[dict[str, Any]] | None = None,
            summary: dict[str, Any] | None = None,
            ml_context: tuple[dict[str, Any] | None, dict[str, Any]] | None = None) -> dict[str, Any]:
    records = records if records is not None else _records(store, user)
    if len(records) < 3:
        return {
            "pass_probability": None, "expected_accuracy": None, "accuracy_range": None,
            "confidence": "none", "model": "insufficient_data", "observations": len(records),
            "message": "Добавьте хотя бы три результата. До этого персональный прогноз не рассчитывается.",
            "factors": [],
        }
    stars = [r["analysis"]["stars"] for r in records]
    weights = [r["weight"] for r in records]
    # A score's accuracy adjusts the player's demonstrated comfortable difficulty.
    abilities = [s + (r["accuracy"] - 0.9) * 3.5 - (0 if r["passed"] else 0.65)
                 for s, r in zip(stars, records)]
    ability = _weighted_quantile(abilities, weights, 0.7)
    prior = _sigmoid((ability - target["stars"]) * 1.65)

    # Similar played maps are the local, data-driven part of the estimate.
    target_vector = np.asarray(vector(target))
    candidates = []
    for r in records:
        v = np.asarray(vector(r["analysis"]))
        scaled = (v - target_vector) / np.asarray([1.3, 1.2, 1.2, 1.8, 2.0, 0.9])
        distance = float(np.sqrt(np.sum(scaled ** 2)))
        w = r["weight"] * math.exp(-0.5 * distance * distance)
        candidates.append((r, w))
    nearby_weight = sum(w for _, w in candidates)
    p = (3 * prior + sum(w * r["passed"] for r, w in candidates)) / (3 + nearby_weight)
    mean_acc = sum(r["accuracy"] * r["weight"] for r in records) / sum(weights)
    baseline_acc = _clamp(mean_acc - (target["stars"] - ability) * 0.042, 0.5, 0.995)
    acc = (2 * baseline_acc + sum(w * _clamp(
        r["accuracy"] - (target["stars"] - r["analysis"]["stars"]) * 0.025, 0.3, 1)
        for r, w in candidates)) / (2 + nearby_weight)

    failed = sum(not r["passed"] for r in records)
    model = "similarity_and_prior"
    ml_p, ml_status = predict_pass(store, user, target, ml_context)
    if ml_p is not None:
        p = 0.7 * ml_p + 0.3 * p
        model = "validated_personal_logistic_and_similarity"
    has_both = failed > 0 and failed < len(records)
    confidence = "medium" if len(records) >= 25 and has_both and nearby_weight >= 2 else "low"
    half_width = 0.065 if confidence == "medium" else 0.105
    factors = []
    delta = target["stars"] - ability
    if delta > 0.35:
        factors.append(f"Сложность карты выше твоего текущего ориентира примерно на {delta:.1f} звезды.")
    elif delta < -0.35:
        factors.append(f"Сложность карты ниже твоего текущего ориентира примерно на {-delta:.1f} звезды.")
    summary = summary or _summary_from_records(user, records)
    if summary["aim_level"] is not None and target["aim"] > summary["aim_level"] + 0.45:
        factors.append("Aim нагрузка выше, чем на большинстве успешно пройденных тобой карт.")
    if summary["speed_level"] is not None and target["speed"] > summary["speed_level"] + 0.45:
        factors.append("Скоростная нагрузка выше, чем на большинстве успешно пройденных тобой карт.")
    if not has_both:
        factors.append("В истории мало неудачных попыток, поэтому вероятность pass особенно ненадёжна.")
    if not factors:
        factors.append("Общая нагрузка близка к твоим недавним результатам.")
    return {
        "pass_probability": round(_clamp(p, 0.02, 0.98) * 100),
        "expected_accuracy": round(acc * 100, 1),
        "accuracy_range": [round(_clamp(acc - half_width, 0, 1) * 100, 1),
                           round(_clamp(acc + half_width, 0, 1) * 100, 1)],
        "confidence": confidence, "model": model, "observations": len(records),
        "ml_status": ml_status,
        "message": "Экспериментальная оценка по вашим результатам; вероятность пока не откалибрована на независимой выборке.",
        "factors": factors,
    }


def recommendations(store: Store, user: str, goal: str = "comfort", mods: str = "NM", limit: int = 12) -> list[dict[str, Any]]:
    ranges = {"warmup": (80, 100), "comfort": (55, 85), "challenge": (25, 60)}
    lo, hi = ranges.get(goal, ranges["comfort"])
    played_ids = {r["map_id"] for r in store.plays(user)}
    records = _records(store, user)
    summary = _summary_from_records(user, records)
    ml_context = model_context(store, user) if len(records) >= 3 else None
    ranked = []
    for item in store.list_maps():
        if item["id"] in played_ids:
            continue
        try:
            analysis = store.analysis(item["id"], mods)
        except (ValueError, RuntimeError):
            continue
        if not analysis:
            continue
        prediction = predict(store, user, analysis, records, summary, ml_context)
        p = prediction["pass_probability"]
        if p is None:
            # With no history, simply sort by chart stars and avoid invented personalization.
            distance = analysis["stars"]
        else:
            distance = abs(p - (lo + hi) / 2) + (25 if p < lo or p > hi else 0)
        ranked.append({**{key: value for key, value in analysis.items() if key not in ("windows", "peaks")},
                       "prediction": prediction, "_distance": distance})
    ranked.sort(key=lambda row: row["_distance"])
    for item in ranked:
        item.pop("_distance")
    return ranked[:limit]
